// Full descriptors for the inference operators supported by the secondary audit.
#ifndef TASO_CANDIDATE_SNAPSHOT_H
#define TASO_CANDIDATE_SNAPSHOT_H
#include <sstream>
#include <iomanip>
namespace taso {
std::string Graph::audit_node_json(Op op, bool weights) {
  std::ostringstream out;
  out << std::setprecision(17);
  OpBase* p = op.ptr;
  bool complete = true;
  out << "{\"guid\":" << op.guid << ",\"type\":" << int(p->type) << ",\"attributes\":[";
  switch (p->type) {
    case OP_CONV2D: { auto v=static_cast<Conv2D*>(p); out<<v->strideH<<','<<v->strideW<<','<<v->padding<<','<<v->activation; break; }
    case OP_POOL2D_MAX: case OP_POOL2D_AVG: { auto v=static_cast<Pool2D*>(p);out<<v->kernelH<<','<<v->kernelW<<','<<v->strideH<<','<<v->strideW<<','<<v->padding<<','<<v->activation;break; }
    case OP_RELU: case OP_SIGMOID: case OP_TANH: out<<static_cast<Activation*>(p)->inPlace;break;
    case OP_CONCAT: { auto v=static_cast<Concat*>(p);out<<v->axis;for(int k=0;k<v->numInputs;k++)out<<','<<v->needCopy[k];break; }
    case OP_SPLIT: { auto v=static_cast<Split*>(p);out<<v->axis;for(int k:v->sizes)out<<','<<k;break; }
    case OP_TRANSPOSE: { auto v=static_cast<Transpose*>(p);out<<v->permIdx<<','<<v->shuffle;break; }
    case OP_MERGE_GCONV: out<<static_cast<MergeGConv*>(p)->count;break;
    case OP_MATMUL: out<<static_cast<Matmul*>(p)->activation;break;
    case OP_BATCHNORM: out<<static_cast<BatchNorm*>(p)->epsilon;break;
    case OP_INPUT: case OP_WEIGHT: case OP_DROPOUT: case OP_RESHAPE:
    case OP_EW_ADD: case OP_EW_MUL: case OP_MUL: case OP_ENLARGE:
    case OP_CONSTANT_IMM: case OP_CONSTANT_ICONV: case OP_CONSTANT_ONE: case OP_CONSTANT_POOL:
    case OP_FUSE_CONV_BATCHNORM: case OP_FUSE_CONV_BATCHNORM_ALPHA_VAR:
    case OP_FUSE_CONV_BATCHNORM_BIAS: case OP_BROADCAST_ADD: break;
    default: complete=false;
  }
  out << "],\"complete\":" << (complete?"true":"false") << ",\"inputs\":[";
  for(int side=0;side<2;side++) {
    const int count = side ? p->numOutputs : p->numInputs;
    for(int k=0;k<count;k++) {
      if(k)out<<',';
      const Tensor& tensor = side ? p->outputs[k] : p->inputs[k];
      int keys[Tensor::MAX_KEY_LENGTH], length=0; tensor.serialize(keys,length);
      out<<'[';for(int j=0;j<length;j++){if(j)out<<',';out<<keys[j];}out<<']';
    }
    if(!side)out<<"],\"outputs\":[";
  }
  out << ']';
#ifdef USE_CUDNN
  if(p->type==OP_CONV2D)out<<",\"backend_algorithm\":"<<int(static_cast<Conv2D*>(p)->fwdAlgo);
#endif
  if(weights && p->type==OP_WEIGHT) {
    std::vector<DATATYPE> values(p->inputs[0].volume());
    model->copy_memory(values.data(), static_cast<const DATATYPE*>(p->inputs[0].data_ptr), values.size()*sizeof(DATATYPE));
    const unsigned char* bytes=reinterpret_cast<const unsigned char*>(values.data());
    const char* hex="0123456789abcdef";
    out<<",\"weight_hex\":\"";
    for(size_t k=0;k<values.size()*sizeof(DATATYPE);k++)out<<hex[bytes[k]>>4]<<hex[bytes[k]&15];
    out<<'"';
  }
  out<<'}'; return out.str();
}
void Graph::audit_write_record(std::ostream& out, std::set<size_t>& seen) {
  for (const auto& entry : inEdges) if (seen.insert(entry.first.guid).second)
    out << "{\"event\":\"node\",\"node\":" << audit_node_json(entry.first,true) << "}\n";
  out << "{\"event\":\"graph\",\"id\":" << audit_id << ",\"nodes\":[";
  bool comma=false;
  for(const auto& entry:inEdges){if(comma)out<<',';comma=true;out<<entry.first.guid;}
  out<<"],\"edges\":[";comma=false;
  for(const auto& entry:inEdges)for(const auto& edge:entry.second){
    if(!edge.srcOp.ptr)continue;if(comma)out<<',';comma=true;
    out<<'['<<edge.srcOp.guid<<','<<edge.srcIdx<<','<<edge.dstOp.guid<<','<<edge.dstIdx<<']';
  }
  out<<"],\"estimated_ms\":"<<total_cost()<<"}\n";
}
std::string Graph::audit_snapshot_json() {
  std::ostringstream out;out<<"{\"native_graph_id\":"<<audit_id<<",\"nodes\":[";bool comma=false;
  for(const auto& entry:inEdges){if(comma)out<<',';comma=true;out<<audit_node_json(entry.first,true);}
  out<<"],\"edges\":[";comma=false;
  for(const auto& entry:inEdges)for(const auto& edge:entry.second){
    if(!edge.srcOp.ptr)continue;if(comma)out<<',';comma=true;
    out<<'['<<edge.srcOp.guid<<','<<edge.srcIdx<<','<<edge.dstOp.guid<<','<<edge.dstIdx<<']';
  }
  out<<"]}";return out.str();
}
}
#endif
