// CPU allocation/ownership backend for the exact production Graph timing source.
#include <cassert>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
using namespace std;
using DATATYPE = float;
const int MAX_DIM=8, MAX_NUM_INPUTS=6, GUID_PRESERVED=99;
enum Type { OP_INPUT,OP_WEIGHT,OP_ANY,OP_CONV2D,OP_DROPOUT,OP_POOL2D_MAX,
 OP_POOL2D_AVG,OP_RELU,OP_SIGMOID,OP_TANH,OP_BATCHNORM,OP_CONCAT,OP_SPLIT,
 OP_RESHAPE,OP_TRANSPOSE,OP_EW_ADD,OP_EW_MUL,OP_MATMUL,OP_MUL,OP_ENLARGE,
 OP_MERGE_GCONV,OP_CONSTANT_IMM,OP_CONSTANT_ICONV,OP_CONSTANT_ONE,OP_CONSTANT_POOL,
 OP_FUSE_CONV_BATCHNORM,OP_FUSE_CONV_BATCHNORM_ALPHA_VAR,OP_FUSE_CONV_BATCHNORM_BIAS,OP_BROADCAST_ADD };
struct Tensor { int numDim=1; int dim[MAX_DIM]={1}; DATATYPE* data_ptr=nullptr; };
class OpBase;
struct Model {
 set<void*> blocks;
 map<int,int> forwards;
 int live=0, guid=500;
 bool fail_measure=false;
 void* allocate_memory(size_t n) { void* p=malloc(n?n:1); blocks.insert(p); return p; }
 void release_memory(void* p) { assert(blocks.erase(p)==1); free(p); }
 float measure_oplist_runtime(const vector<OpBase*>& ops);
};
class OpBase {
public:
 Model* model; int type; Tensor inputs[MAX_NUM_INPUTS],outputs[2];
 int strideH=1,strideW=1,padding=0,activation=0,permIdx=0,count=1,kernelH=1,kernelW=1,axis=0;
 float epsilon=.1; bool shuffle=false,inPlace=false,needCopy[MAX_NUM_INPUTS]={};
 vector<int> sizes{1}; bool owns=false;
 OpBase(Model* m,int t):model(m),type(t) { ++model->live; }
 virtual ~OpBase() { --model->live; }
 virtual void map() { outputs[0].data_ptr=(DATATYPE*)model->allocate_memory(sizeof(float));owns=true; }
 virtual void unmap() { if(owns)model->release_memory(outputs[0].data_ptr);owns=false; }
 virtual void forward(bool block=false) { ++model->forwards[type]; }
};
class NoOp:public OpBase {
public: NoOp(Model* m,Tensor x,int t):OpBase(m,t) { inputs[0]=outputs[0]=x; }
 void map() override {} void unmap() override {} void forward(bool=false) override {}
};
class Activation:public OpBase {
public: Activation(Model*m,Tensor x,int t,bool inplace):OpBase(m,t) { inputs[0]=x;inPlace=inplace; }
 void map() override { if(inPlace)outputs[0]=inputs[0];else OpBase::map(); }
};
class Conv2D:public OpBase {
public: Conv2D(Model*m,Tensor x,Tensor w,int,int,int,int):OpBase(m,OP_CONV2D) {inputs[0]=x;inputs[1]=w;}
};
class Constant:public OpBase {
public: Constant(Model*m,int,int*,int t):OpBase(m,t) {}
};
#define STUB(C,T) class C:public OpBase {public:template<class... A>C(Model*m,A...):OpBase(m,T) {}};
STUB(Matmul,OP_MATMUL) STUB(Reshape,OP_RESHAPE) STUB(Transpose,OP_TRANSPOSE)
STUB(Element,OP_EW_ADD) STUB(Enlarge,OP_ENLARGE) STUB(MergeGConv,OP_MERGE_GCONV)
STUB(Mul,OP_MUL) STUB(Pool2D,OP_POOL2D_MAX) STUB(BatchNorm,OP_BATCHNORM)
STUB(Split,OP_SPLIT) STUB(Concat,OP_CONCAT) STUB(FuseConvBatchNorm,OP_FUSE_CONV_BATCHNORM)
STUB(FuseConvBatchNormAlphaVar,OP_FUSE_CONV_BATCHNORM_ALPHA_VAR)
STUB(FuseConvBatchNormBias,OP_FUSE_CONV_BATCHNORM_BIAS) STUB(BroadcastAdd,OP_BROADCAST_ADD)
struct Op {size_t guid;OpBase* ptr;};
struct OpCompare {bool operator()(Op a,Op b)const{return a.guid<b.guid;}};
struct Edge {Op srcOp,dstOp;int srcIdx,dstIdx;};
struct EdgeCompare {bool operator()(const Edge&a,const Edge&b)const {
 return tie(a.srcOp.guid,a.dstOp.guid,a.srcIdx,a.dstIdx)<tie(b.srcOp.guid,b.dstOp.guid,b.srcIdx,b.dstIdx);}};
struct Graph {
 Model* model; map<Op,set<Edge,EdgeCompare>,OpCompare> inEdges,outEdges;
 float run();float run_time_for_feedback();float run_runtime(bool);
 void edge(Op src,Op dst,int port=0) { Edge e{src,dst,0,port};inEdges[dst].insert(e);if(src.ptr)outEdges[src].insert(e); }
};
float Model::measure_oplist_runtime(const vector<OpBase*>& ops) {
 for(auto op:ops) {
  assert(op->type!=OP_WEIGHT && op->type!=OP_CONSTANT_ONE && op->type!=OP_MUL && op->type!=OP_ENLARGE);
  if(op->type==OP_CONV2D)assert(op->inputs[0].data_ptr && op->inputs[1].data_ptr);
  op->forward();
 }
 if(fail_measure)throw runtime_error("timing failure");
 return .5f;
}
