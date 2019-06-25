#ifndef tools_HPP_INCLUDED
#define tools_HPP_INCLUDED

#include <string>
void genuid(std::string* id);
#define gpuipc "ipc:///tmp/gSMFRETda"

#include <boost/histogram.hpp>

#include <vector>
using namespace boost::histogram;
using reg = axis::regular<>;
namespace boost {
namespace histogram {

template <typename... Ts>
auto make_axis_vector(Ts&&... ts) {
  using T = axis::variant<detail::remove_cvref_t<Ts>...>;
  return std::vector<T>({T(std::forward<Ts>(ts))...});
}

using static_tag = std::false_type;
using dynamic_tag = std::true_type;

template <typename... Axes>
auto make(static_tag, Axes&&... axes) {
  return make_histogram(std::forward<Axes>(axes)...);
}

template <typename S, typename... Axes>
auto make_s(static_tag, S&& s, Axes&&... axes) {
  return make_histogram_with(s, std::forward<Axes>(axes)...);
}

template <typename... Axes>
auto make(dynamic_tag, Axes&&... axes) {
  return make_histogram(make_axis_vector(std::forward<Axes>(axes)...));
}

template <typename S, typename... Axes>
auto make_s(dynamic_tag, S&& s, Axes&&... axes) {
  return make_histogram_with(s, make_axis_vector(std::forward<Axes>(axes)...));
}

} // namespace histogram
} // namespace boost

#endif