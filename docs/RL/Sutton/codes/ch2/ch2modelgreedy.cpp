/*
 * ch2modelgreedy.cpp
 *
 *  Created on: Dec 9, 2020
 *      Author: zf
 */



#include "ch2modelgreedy.h"
#include <torch/torch.h>
#include <cstdlib>
#include <iostream>
#include <ctime>

Ch2GreedyModel::Ch2GreedyModel(int num, float e)
	: Ch2Model(num), epsilon(e), counters(num, 0), probTensor(torch::ones({1}) * e){
	values = torch::zeros({actionNum});
	std::srand (std::time(NULL));
}

int Ch2GreedyModel::getAction() {
	torch::Tensor greedyOutTensor = torch::binomial(countTensor, probTensor);
	int greedyOutput = greedyOutTensor.item().toInt();

	if (greedyOutput == 1) { //==1
		return std::rand() % actionNum;
	} else {
		return values.argmax().item().toInt();
	}
}

void Ch2GreedyModel::update(int action, float reward) {
	counters[action] ++;
	const int n = counters[action];
	const float value = values[action].item().toFloat();

	values[action] += (reward - value) / n;
}
