"""Run small official Attention/ViT shapes with CPU logical kernel substitutes.
This does not verify CUDA quantization numerics. The implementation directory must
be a local copy of the pinned official inference/ directory documented in docs/deepseek.
"""
import argparse
import sys, types, torch
parser=argparse.ArgumentParser()
parser.add_argument('--reference-dir',required=True)
options=parser.parse_args()
sys.path.insert(0,options.reference_dir)
k=types.ModuleType('kernel')
def quant(x,*args,**kwargs): return x
# CPU logical sparse-attention replacement, ignores CUDA packing but implements tensor math.
def sparse(q,kv,sink,indices,scale):
 b,s,h,d=q.shape
 selected=kv[torch.arange(b)[:,None,None],indices.clamp(min=0)]
 scores=torch.einsum('bshd,bskd->bshk',q.float(),selected.float())*scale
 scores=scores.masked_fill(indices[:,:,None,:]<0,-torch.inf)
 scores=torch.cat([scores,sink[None,None,:,None].expand(b,s,-1,-1)],dim=-1)
 weights=scores.softmax(-1)[...,:-1]
 return torch.einsum('bshk,bskd->bshd',weights,selected.float()).to(q.dtype)
for name in ['act_quant','fp4_act_quant']:setattr(k,name,quant)
for name in ['fp4_gemm','fp8_gemm','hc_split_sinkhorn']:setattr(k,name,lambda *a,**kw:None)
k.sparse_attn=sparse;sys.modules['kernel']=k
import model
model.default_dtype=torch.float32
args=model.ModelArgs(dim=64,n_heads=4,q_lora_rank=32,head_dim=32,rope_head_dim=16,o_groups=2,o_lora_rank=16,window_size=4,n_layers=8,n_mtp_layers=0,compress_ratios=(0,2,2,2,1,1,1,1),kv_source_layers=(1,4),index_source_layers=(1,4,6),index_n_heads=2,index_head_dim=32,index_topk=3,candidate_source_layer=4,candidate_topk_blocks=2,candidate_block_size=2,max_batch_size=2,max_seq_len=32)
attentions=[model.Attention(i,args).float() for i in range(8)]
for a in attentions:
 for p in a.parameters():torch.nn.init.normal_(p,std=.02)
reports=[]
for start,length in [(0,7),(7,1),(8,1)]:
 for i,a in enumerate(attentions):
  x=torch.randn(2,length,64);out=a(x,start)
  assert list(out.shape)==[2,length,64]
  if args.compress_ratios[i]:
   c=(start+length)//args.compress_ratios[i]
   assert list(model.shared_attn.compress_kv[:2,:c].shape)==[2,c,32]
   assert list(model.shared_attn.topk_idxs.shape)==[2,length,min(3,c)]
  reports.append({'start':start,'length':length,'layer':i,'output':list(out.shape),'topk':list(model.shared_attn.topk_idxs.shape) if args.compress_ratios[i] else None})
print('Official Attention forward (CPU logical kernel substitute):',len(reports),'layer calls verified')
from vision import ViT,Aligner
va=model.ModelArgs(dim=64,vision_n_layers=2,vision_dim=32,vision_n_heads=4,vision_inter_dim=64,vision_patch_size=2,vision_downsample_ratio=3)
v=ViT(va);align=Aligner(va)
y=v(torch.randn(20,3,2,2),4,5);z=align(y,4,5)
assert list(y.shape)==[20,32] and list(z.shape)==[4,64]
print('Official ViT / Aligner: non-divisible grid 4 x 5 ->',list(y.shape),'->',list(z.shape))
