#include "taso/search_schedule.h"
#include <cassert>
#include <queue>
#include <iostream>
int main() {
  using namespace taso;
  std::vector<int> budgets{2, 3};
  std::vector<std::vector<int> > rules{{0, 3}, {}};
  SearchSchedule schedule(budgets, rules);
  std::priority_queue<int> frontier;
  frontier.push(10);
  std::vector<size_t> phases;
  size_t phase;
  // Candidates survive the transition: second phase receives work generated
  // in the first phase; it does not restart at the input.
  while (!frontier.empty() && schedule.next(phase)) {
    int candidate = frontier.top(); frontier.pop();
    phases.push_back(phase);
    frontier.push(candidate - 1);
  }
  assert((phases == std::vector<size_t>{0, 0, 1, 1, 1}));
  assert(frontier.top() == 5 && schedule.total() == 5);
  assert(schedule.rules(0).size() == 2 && schedule.rules(1).empty());
  assert(!schedule.next(phase));
  bool rejected = false;
  try { SearchSchedule invalid({0}, {{}}); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { SearchSchedule invalid({INT_MAX, 1}, {{}, {}}); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  SearchSchedule exhausted({5, 5}, {{}, {}});
  assert(exhausted.next(phase) && exhausted.total() == 1);
  // A caller stops on an empty frontier without consuming more expansions.
  assert(exhausted.total() == 1);
  std::cout << "Native phase-budget and queue-continuity tests passed\n";
}
