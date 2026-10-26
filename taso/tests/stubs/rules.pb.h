// Declarations only: lets syntax checks cover core code without installing
// protobuf/CUDA on a Mac. GPU builds use protoc's real generated header.
#ifndef TASO_TEST_RULES_STUB
#define TASO_TEST_RULES_STUB
#include <istream>
#define GOOGLE_PROTOBUF_VERIFY_VERSION do {} while (0)
namespace GraphSubst {
class Parameter { public: int key() const; int value() const; };
class Tensor { public: int opid() const; int tsid() const; };
class Operator { public: int type() const; int input_size() const; const Tensor& input(int) const; int para_size() const; const Parameter& para(int) const; };
class MapOutput { public: int srcopid() const; int dstopid() const; int srctsid() const; int dsttsid() const; };
class Rule { public: int srcop_size() const; const Operator& srcop(int) const; int dstop_size() const; const Operator& dstop(int) const; int mappedoutput_size() const; const MapOutput& mappedoutput(int) const; };
class RuleCollection { public: bool ParseFromIstream(std::istream*); int rule_size() const; const Rule& rule(int) const; };
}
#endif
