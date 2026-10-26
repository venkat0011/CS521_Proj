// Phase budget accounting is independent of the CUDA/TASO runtime.
#ifndef TASO_SEARCH_SCHEDULE_H
#define TASO_SEARCH_SCHEDULE_H
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace taso {
class SearchSchedule {
public:
  SearchSchedule(const std::vector<int>& budgets,
                 const std::vector<std::vector<int> >& rules)
      : budgets_(budgets), rules_(rules), phase_(0), used_(0), total_(0) {
    if (budgets.empty() || budgets.size() != rules.size())
      throw std::invalid_argument("Schedule needs equally sized, nonempty budget/rule lists");
    long long sum = 0;
    for (size_t i = 0; i < budgets.size(); ++i) {
      if (budgets[i] <= 0) throw std::invalid_argument("Phase budgets must be positive");
      sum += budgets[i];
      if (sum > INT_MAX) throw std::invalid_argument("Total budget exceeds INT_MAX");
    }
  }
  bool next(size_t& phase) {
    if (phase_ < budgets_.size() && used_ == budgets_[phase_]) {
      ++phase_;
      used_ = 0;
    }
    if (phase_ == budgets_.size()) return false;
    phase = phase_;
    ++used_;
    ++total_;
    return true;
  }
  const std::vector<int>& rules(size_t phase) const { return rules_.at(phase); }
  int total() const { return total_; }
private:
  std::vector<int> budgets_;
  std::vector<std::vector<int> > rules_;
  size_t phase_;
  int used_, total_;
};
}  // namespace taso
#endif
