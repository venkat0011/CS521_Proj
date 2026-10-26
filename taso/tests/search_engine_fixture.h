// Synthetic graph backend for testing the production search loop without CUDA.
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include "taso/search_schedule.h"
using namespace std;
namespace taso {
enum PaddingMode { PD_MODE_SAME, PD_MODE_VALID };
enum ActiMode { AC_MODE_NONE, AC_MODE_RELU };
struct Model {};
struct Op { string to_string() const { return "fixture"; } };
struct OpX {};
struct RewriteStats {
 size_t attempts=0, match_checks=0, full_matches=0, operator_rejections=0;
 size_t output_rejections=0, generated=0, cycle_rejections=0;
 size_t cost_rejections=0, size_rejections=0, duplicates=0, admitted=0;
 double seconds=0;
};
class Graph {
public:
 static int live;
 Model* model = NULL;
 int id;
 float cost;
 map<int,int> inEdges;
 struct GraphSubst { vector<Op> srcOps, dstOps; };
 vector<GraphSubst> subst_history;
 Graph(int i=0, float c=10) : id(i), cost(c) { ++live; inEdges[0]=0; inEdges[1]=0; }
 ~Graph() { --live; }
 float total_cost() { return cost; }
 size_t hash() { return id; }
 void print_costs() {}
 Graph* preprocess_weights() { return new Graph(id,cost); }
 Graph* optimize(float, int, bool);
 Graph* optimize_scheduled(float,const vector<int>&,const vector<vector<int> >&,bool,const string&);
};
int Graph::live=0;
struct GraphCompare { bool operator()(Graph* a,Graph* b) { return a->cost > b->cost; } };
class GraphXfer {
public:
 static int next_id;
 int id;
 vector<OpX*> srcOps,dstOps;
 GraphXfer() : id(next_id++ % 18) {}
 static GraphXfer* create_conv_relu(Model*,int,int,PaddingMode) { return new GraphXfer; }
 static GraphXfer* create_conv_batch(Model*,int,int,PaddingMode) { return new GraphXfer; }
 static GraphXfer* create_conv_mul(Model*,int,int,PaddingMode) { return new GraphXfer; }
 static GraphXfer* create_enlarge_merge_convs(Model*,ActiMode) { return new GraphXfer; }
 static GraphXfer* create_merge_group_convs(Model*,int,int,ActiMode) { return new GraphXfer; }
 static void load_graph_xfer_from_pb_file(Model*,vector<GraphXfer*>&,string) {}
 void run(int,Graph* graph,priority_queue<Graph*,vector<Graph*>,GraphCompare>& queue,
          set<size_t>& seen,float threshold,int,RewriteStats* stats) {
   if (stats) ++stats->match_checks;
   int child=-1; float cost=0;
   if (graph->id==0 && id==0) { child=1; cost=8; }
   if (graph->id==0 && id==1) { child=2; cost=9; }
   if (graph->id==1 && id==1) { child=3; cost=5; }
   if (graph->id==2 && id==0) { child=4; cost=4; }
   if (child<0) return;
   if (stats) { ++stats->full_matches; ++stats->generated; }
   if (cost>=threshold) { if(stats)++stats->cost_rejections; return; }
   if (!seen.insert(child).second) { if(stats)++stats->duplicates; return; }
   queue.push(new Graph(child,cost));
   if(stats)++stats->admitted;
 }
};
int GraphXfer::next_id=0;
}
using namespace taso;
