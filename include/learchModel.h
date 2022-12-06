//
// Created by prophe on 2022/12/2.
//

#ifndef NUMPYLEARCH_LEARCHMODEL_H
#define NUMPYLEARCH_LEARCHMODEL_H

#include <vector>
#include <string>
#include "NumCpp.hpp"
using namespace std;

class PolicyFeedForwardNp {
private:
    nc::NdArray<double> mean; // (1, 47)
    nc::NdArray<double> scale; // (1, 47)

    nc::NdArray<double> linear1; // (47, 64)
    nc::NdArray<double> bias1; // (1, 64)

    nc::NdArray<double> linear2; // (64, 64)
    nc::NdArray<double> bias2; // (1, 64)

    nc::NdArray<double> linear3; // (64, 1)
    nc::NdArray<double> bias3; // (1, 1)

public:
    PolicyFeedForwardNp(string modelPath) {
        mean = nc::fromfile<double>(modelPath + "/mean", '\n');
        scale = nc::fromfile<double>(modelPath + "/scale", '\n');

        bias1 = nc::fromfile<double>(modelPath + "/bias1", '\n');
        linear1 = nc::fromfile<double>(modelPath + "/linear1", '\n');
        linear1.reshape(47, 64);

        bias2 = nc::fromfile<double>(modelPath + "/bias2", '\n');
        linear2 = nc::fromfile<double>(modelPath + "/linear2", '\n');
        linear2.reshape(64, 64);

        bias3 = nc::fromfile<double>(modelPath + "/bias3", '\n');
        linear3 = nc::fromfile<double>(modelPath + "/linear3", '\n');
        linear3.reshape(64, 1);
    }

    nc::NdArray<double> relu(nc::NdArray<double> x) {
        return nc::where(x >= 0.0, x, 0.0);
    }

    nc::NdArray<double> forward(vector<double> vecs) {
        nc::NdArray<double> x(vecs);
        x.reshape(1, -1);
        nc::NdArray<double> out = nc::divide(x - mean, scale);
        out = relu(nc::add(nc::matmul(out, linear1), bias1));
        out = relu(nc::add(nc::matmul(out, linear2), bias2));
        out = nc::add(nc::matmul(out, linear3), bias3);
        return out;
    }

    nc::NdArray<double> forward_batch(vector<vector<double> > vecs) {
        nc::NdArray<double> x(vecs); // shape = [num_state, 47]
        nc::NdArray<double> out = nc::divide(x - mean.repeat(vecs.size(), 1), scale.repeat(vecs.size(), 1)); // shape = [num_state, 47]
        out = relu(nc::add(nc::matmul(out, linear1), bias1.repeat(vecs.size(), 1)));
        out = relu(nc::add(nc::matmul(out, linear2), bias2.repeat(vecs.size(), 1)));
        out = nc::add(nc::matmul(out, linear3), bias3.repeat(vecs.size(), 1));
        return out;
    }
};

#endif //NUMPYLEARCH_LEARCHMODEL_H
