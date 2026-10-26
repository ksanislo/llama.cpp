#pragma once

// LLAMA_LAYER_SPLIT: layer split by prompt cost instead of free memory / -ts (a piped prompt runs at its slowest stage).
// even: computed layers weigh 1; cost: recurrent 1, attention LLAMA_LAYER_SPLIT_ATTN (1.8); a,b,..: layer counts per device.
// even/cost: contiguous ranges, smallest maximum stage weight, earlier stages as light as that allows.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

// device of every layer for weights w over n_dev stages; empty if n_dev < 1
static std::vector<int> llama_layer_split_balance(const std::vector<double> & w, int n_dev) {
    const int n = (int) w.size();
    if (n_dev < 1) {
        return {};
    }
    std::vector<int> dev(n, 0);
    if (n == 0) {
        return dev;
    }
    if (n <= n_dev) {
        for (int i = 0; i < n; ++i) {
            dev[i] = i;
        }
        return dev;
    }
    std::vector<double> pre(n + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        pre[i + 1] = pre[i] + std::max(0.0, w[i]);
    }
    auto seg = [&](int a, int b) { return pre[b] - pre[a]; }; // layers [a, b)

    // best[d][i]: smallest max weight of the first i layers over d+1 stages, each with >= 1 layer
    const double inf = 1e300;
    std::vector<std::vector<double>> best(n_dev, std::vector<double>(n + 1, inf));
    for (int i = 1; i <= n; ++i) {
        best[0][i] = seg(0, i);
    }
    for (int d = 1; d < n_dev; ++d) {
        for (int i = d + 1; i <= n; ++i) {
            double b = inf;
            for (int j = d; j < i; ++j) {
                b = std::min(b, std::max(best[d - 1][j], seg(j, i)));
            }
            best[d][i] = b;
        }
    }
    const double M   = best[n_dev - 1][n];
    const double eps = 1e-9 * std::max(1.0, M);

    // can layers [a, n) go on g stages (each >= 1 layer) with no stage above M?
    auto fits = [&](int a, int g) {
        if (n - a < g) {
            return false;
        }
        int need = 0;
        for (int i = a; i < n; ) {
            int j = i + 1;
            if (seg(i, j) > M + eps) {
                return false;
            }
            while (j < n && seg(i, j + 1) <= M + eps) {
                ++j;
            }
            ++need;
            i = j;
        }
        return need <= g;
    };

    int a = 0;
    for (int d = 0; d < n_dev - 1; ++d) {
        const int k_max = n - a - (n_dev - 1 - d); // leaves one layer per later stage
        int k = 1;
        while (k < k_max && !(seg(a, a + k) <= M + eps && fits(a + k, n_dev - 1 - d))) {
            ++k;
        }
        for (int i = a; i < a + k; ++i) {
            dev[i] = d;
        }
        a += k;
    }
    for (int i = a; i < n; ++i) {
        dev[i] = n_dev - 1;
    }
    return dev;
}

// "a,b,c": layer counts per device -> device of every layer; empty with err set if malformed
static std::vector<int> llama_layer_split_counts(const char * s, int n, int n_dev, std::string & err) {
    std::vector<int> cnt;
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > 100000) {
            err = "expected comma-separated layer counts";
            return {};
        }
        cnt.push_back((int) v);
        p = end;
        if (*p == ',' && p[1]) {
            ++p;
        } else if (*p) {
            err = "expected comma-separated layer counts";
            return {};
        }
    }
    if ((int) cnt.size() != n_dev) {
        err = "needs one count per device (" + std::to_string(n_dev) + ")";
        return {};
    }
    long sum = 0;
    for (int c : cnt) {
        sum += c;
    }
    if (sum != n) {
        err = "the counts add up to " + std::to_string(sum) + ", the model has " + std::to_string(n) + " layers";
        return {};
    }
    std::vector<int> dev;
    dev.reserve(n);
    for (int d = 0; d < n_dev; ++d) {
        dev.insert(dev.end(), cnt[d], d);
    }
    return dev;
}
