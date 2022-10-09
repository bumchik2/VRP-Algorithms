#include <iostream>
#include "../src/utils/time_utils.h"

using std::cout;
using std::endl;

int main() {
    const auto res = time_window_to_begin_seconds_end_seconds("02:15:47-2.14:50:00");
    cout << res.first << ' ' << res.second << endl;
    return 0;
}
