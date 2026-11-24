#include "taso/ops.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
using namespace taso;
namespace taso {
Model::Model():global_unique_id(100){}
void Matmul::set_layout(){}
void* Model::allocate_memory(size_t n,const float* p){void* out=malloc(n);if(p)memcpy(out,p,n);return out;}
void Model::release_memory(void* p){free(p);}
bool Model::copy_memory(float* dst,const float* src,size_t n){memcpy(dst,src,n);return true;}
void Model::measure_conv2d_cost(Conv2D* op){op->runtime=1;}
#define BACKEND(T) void T::map(){} void T::unmap(){} void T::forward(bool){}
BACKEND(Conv2D) BACKEND(Pool2D) BACKEND(Activation) BACKEND(Concat)
BACKEND(Matmul) BACKEND(Reshape) BACKEND(Transpose) BACKEND(Element)
}
std::vector<int> metadata(const Tensor& t){int keys[Tensor::MAX_KEY_LENGTH],n=0;t.serialize(keys,n);return std::vector<int>(keys,keys+n);}
int main(){
 Graph original;int dims[]={1,1,1,1},wdims[]={2,1,1,1};float weights[]={1,-2};
 auto input=original.new_input(4,dims);auto weight=original.new_weight(4,wdims,weights);
 auto conv=original.conv2d(input,weight,1,1,PD_MODE_SAME,AC_MODE_NONE);
 // Reproduce production weight folding: replace_node updates cached inputs,
 // but the cached operator's outputs retain the earlier split annotations.
 Tensor folded=*weight;folded.data_ptr=weights;
 folded.split[0].num=1;folded.split[0].pos[0]=1;
 auto replacement=original.new_weight(folded);
 original.replace_node(weight->op,replacement->op);original.remove_node(weight->op);weight=replacement;
 Graph normalized;
 auto ni=normalized.audit_leaf(OP_INPUT,metadata(input->op.ptr->outputs[0]),{});
 auto nw=normalized.audit_leaf(OP_WEIGHT,metadata(weight->op.ptr->outputs[0]),{1,-2});
 auto nc=normalized.conv2d(ni,nw,1,1,PD_MODE_SAME,AC_MODE_NONE);
 assert(metadata(nc->op.ptr->outputs[0])!=metadata(conv->op.ptr->outputs[0]));
 auto saved=original.audit_snapshot_json();
 auto cache_size=original.model->conv2d.size();auto original_ptr=conv->op.ptr;
 Graph imported;
 auto a=imported.audit_leaf(OP_INPUT,metadata(input->op.ptr->outputs[0]),{});
 auto b=imported.audit_leaf(OP_WEIGHT,metadata(weight->op.ptr->outputs[0]),{1,-2});
 auto c=imported.audit_operator(OP_CONV2D,{a,b},{1,1,0,0},
   {metadata(conv->op.ptr->inputs[0]),metadata(conv->op.ptr->inputs[1])},
   {metadata(conv->op.ptr->outputs[0])},-1);
 assert(c.size()==1 && c[0]->op.ptr!=original_ptr && imported.model->conv2d.size()==cache_size);
 assert(original.audit_snapshot_json()==saved);
 assert(metadata(c[0]->op.ptr->outputs[0])==metadata(conv->op.ptr->outputs[0]));
 assert(metadata(c[0]->op.ptr->inputs[0])==metadata(conv->op.ptr->inputs[0]));
 auto bad=metadata(conv->op.ptr->outputs[0]);bad[2]=2;
 try{imported.audit_operator(OP_CONV2D,{a,b},{1,1,0,0},
   {metadata(conv->op.ptr->inputs[0]),metadata(conv->op.ptr->inputs[1])},{bad},-1);assert(false);}
 catch(const std::runtime_error&){}
 std::cout<<saved<<"\n"<<imported.audit_snapshot_json()<<"\n"<<normalized.audit_snapshot_json()<<"\n";
}
