// Opt-in secondary-experiment journal. Never used for ranking or deduplication.
#ifndef TASO_CANDIDATE_AUDIT_H
#define TASO_CANDIDATE_AUDIT_H
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
namespace taso {
class CandidateAudit {
public:
  std::ofstream stream;
  std::set<size_t> nodes, graphs;
  size_t next_id = 0;
  int expansion = 0, phase = 0, rule = -1;
  static CandidateAudit*& active() { static thread_local CandidateAudit* value = nullptr; return value; }
  explicit CandidateAudit(const std::string& trace) {
    const char* enabled = std::getenv("TASO_CANDIDATE_AUDIT");
    if (!enabled || std::string(enabled) != "1") return;
    if (active()) throw std::runtime_error("Nested candidate audit is not supported");
    if (trace.empty()) throw std::runtime_error("Candidate audit requires a trace path");
    stream.open((trace + ".candidates.jsonl").c_str());
    if (!stream) throw std::runtime_error("Cannot open candidate journal");
    stream << std::setprecision(12);
    active() = this;
  }
  ~CandidateAudit() { if (active() == this) active() = nullptr; }
  void ref(const char* event, Graph* graph, Graph* parent = nullptr) {
    if (!stream.is_open()) return;
    stream << "{\"event\":\"" << event << "\",\"graph\":" << graph->audit_id
      << ",\"parent\":" << (parent ? parent->audit_id : 0)
      << ",\"expansion\":" << expansion << ",\"phase\":" << phase << ",\"rule\":" << rule << "}\n";
    if (!stream) throw std::runtime_error("Candidate journal write failed");
  }
  void capture(const char* event, Graph* graph, Graph* parent = nullptr) {
    if (!stream.is_open()) return;
    if (!graph->audit_id) graph->audit_id = ++next_id;
    if (graphs.insert(graph->audit_id).second) {
      graph->audit_write_record(stream, nodes);
    }
    ref(event, graph, parent);
  }
};
}
#endif
