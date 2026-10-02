#!/usr/bin/env python3
"""End-to-end UMI forward dynamics with ONNX Runtime CUDA modules."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
from types import SimpleNamespace
import onnx,torch
from torch import nn
from diffusers import Cosmos3OmniPipeline,CosmosActionCondition
from diffusers.utils import load_image,export_to_video
from cosmos3edge_inverse_onnx import make_session,bind_input,_DeterministicDistribution,build_pipeline_shell,load_pipeline_configs

class OrtWanCodec(nn.Module):
 def __init__(self,encoder,decoder,config,before_decode=None,release_encoder_after_encode=False):
  super().__init__(); self.enc=make_session(encoder); self.dec=make_session(decoder); self.config=config; self.dtype=torch.float16
  self.before_decode=before_decode; self.release_encoder_after_encode=release_encoder_after_encode
  self.register_parameter('_device_anchor',nn.Parameter(torch.empty(0,device='cuda'),requires_grad=False))
  self.register_buffer('mean',torch.tensor(config.latents_mean).view(1,-1,1,1,1),persistent=False)
  self.register_buffer('std',torch.tensor(config.latents_std).view(1,-1,1,1,1),persistent=False)
 def encode(self,video,return_dict=True):
  video=video.half().cuda().contiguous(); shape=(video.shape[0],48,(video.shape[2]-1)//4+1,video.shape[3]//16,video.shape[4]//16)
  z=torch.empty(shape,dtype=torch.float16,device='cuda'); io=self.enc.io_binding(); keep=bind_input(io,'video_rgb_minus1_to1',video)
  io.bind_output('normalized_latents','cuda',0,onnx.TensorProto.FLOAT16,shape,z.data_ptr()); self.enc.run_with_iobinding(io); del keep
  raw=z*self.std.to(z)+self.mean.to(z)
  if self.release_encoder_after_encode:
   self.enc=None; gc.collect(); torch.cuda.empty_cache()
  out=SimpleNamespace(latent_dist=_DeterministicDistribution(raw)); return out if return_dict else (out.latent_dist,)
 def decode(self,raw,return_dict=True):
  if self.before_decode is not None:
   callback,self.before_decode=self.before_decode,None; callback(); gc.collect(); torch.cuda.empty_cache()
  z=((raw.half()-self.mean.to(raw))/self.std.to(raw)).contiguous(); shape=(z.shape[0],3,(z.shape[2]-1)*4+1,z.shape[3]*16,z.shape[4]*16)
  video=torch.empty(shape,dtype=torch.float16,device='cuda'); io=self.dec.io_binding(); keep=bind_input(io,'normalized_latents',z)
  io.bind_output('video_rgb_minus1_to1','cuda',0,onnx.TensorProto.FLOAT16,shape,video.data_ptr()); self.dec.run_with_iobinding(io); del keep
  out=SimpleNamespace(sample=video); return out if return_dict else (video,)

class OrtForwardTransformer(nn.Module):
 def __init__(self,model,config):
  super().__init__(); self.session=make_session(model); self.config=config; self.dtype=torch.float16; self.action_dim=config.action_dim
  self.register_parameter('_device_anchor',nn.Parameter(torch.empty(0,device='cuda'),requires_grad=False))
 def forward(self,*,input_ids,position_ids,vision_tokens,action_tokens,vision_timesteps,**kw):
  vals={'input_ids':input_ids.long().cuda(),'position_ids':position_ids.float().cuda(),'vision_latents':vision_tokens[0].half().cuda(),
        'action_latents':action_tokens[0].half().cuda(),'vision_timesteps':vision_timesteps.long().cuda()}
  output=torch.empty_like(vals['vision_latents']); io=self.session.io_binding(); keep=[bind_input(io,n,v) for n,v in vals.items()]
  io.bind_output('vision_velocity','cuda',0,onnx.TensorProto.FLOAT16,tuple(output.shape),output.data_ptr()); self.session.run_with_iobinding(io); del keep
  result=([output.to(vision_tokens[0].dtype)],None,[torch.zeros_like(action_tokens[0])]); return result if not kw.get('return_dict',True) else SimpleNamespace(sample=result[0],sound=None,action=result[2])

def main():
 parser=argparse.ArgumentParser()
 for name in ['checkpoint','image','actions','vae-encoder','vae-decoder','forward-transformer','output-video','output-json']:
  parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--max-chunks',type=int,default=None,help='Limit the autoregressive rollout; default: all chunks')
 args=parser.parse_args(); spec=json.loads(args.actions.read_text())
 t0=time.perf_counter(); tc,vc=load_pipeline_configs(args.checkpoint)
 pipe=build_pipeline_shell(args.checkpoint,OrtForwardTransformer(args.forward_transformer,tc),OrtWanCodec(args.vae_encoder,args.vae_decoder,vc)); load_s=time.perf_counter()-t0

 chunks=spec['action_chunks'][:args.max_chunks]
 current_image=load_image(str(args.image)); stitched=[]; chunk_stats=[]
 torch.cuda.synchronize(); rollout_t0=time.perf_counter()
 for index,raw_chunk in enumerate(chunks):
  condition=CosmosActionCondition(mode='forward_dynamics',chunk_size=spec['action_chunk_size'],domain_name=spec['domain_name'],resolution_tier=spec['image_size'],
   raw_actions=torch.tensor(raw_chunk,dtype=torch.float32),image=current_image,view_point=spec['view_point'])
  torch.cuda.synchronize(); chunk_t0=time.perf_counter()
  with torch.inference_mode():
   result=pipe(prompt=spec['prompt'],action=condition,fps=spec['fps'],num_inference_steps=30,guidance_scale=1.0,
    generator=torch.Generator(device='cuda').manual_seed(index),use_system_prompt=False,output_type='pil',enable_safety_check=False)
  torch.cuda.synchronize(); chunk_s=time.perf_counter()-chunk_t0
  if len(result.video)!=spec['action_chunk_size']+1:
   raise RuntimeError(f'chunk {index}: expected {spec["action_chunk_size"]+1} frames, got {len(result.video)}')
  stitched.extend(result.video[1:]); current_image=result.video[-1]
  chunk_stats.append({'chunk':index,'seed':index,'frames_with_conditioning':len(result.video),'generated_frames':len(result.video)-1,'inference_s':chunk_s})
 rollout_s=time.perf_counter()-rollout_t0
 args.output_video.parent.mkdir(parents=True,exist_ok=True)
 export_to_video(stitched,str(args.output_video),fps=spec['fps'],macro_block_size=1)
 payload={'backend':'onnxruntime_cuda_fp16','mode':'forward_dynamics','domain':spec['domain_name'],'chunks':len(chunks),'frames':len(stitched),
  'fps':spec['fps'],'load_s':load_s,'inference_s':rollout_s,'chunk_stats':chunk_stats,'output_video':str(args.output_video)}
 args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))


if __name__ == '__main__':
 main()
