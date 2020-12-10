/*
 * ch2modelconst.cpp
 *
 *  Created on: Dec 9, 2020
 *      Author: zf
 */


#include "ch2modelconst.h"

Ch2ConstModel::Ch2ConstModel(int num, float inAlpha, float e)
	: Ch2Model(num), epsilon(e), alpha(inAlpha), probTensor(torch::ones({1}) * e){
	values = torch::zeros({actionNum});
	std::srand (std::time(NULL));
}

int Ch2ConstModel::getAction() {
	torch::Tensor greedyOutTensor = torch::binomial(countTensor, probTensor);
	int greedyOutput = greedyOutTensor.item().toInt();

	if (greedyOutput == 1) { //==1
		return std::rand() % actionNum;
	} else {
		return values.argmax().item().toInt();
	}
}

void Ch2ConstModel::update(int action, float reward) {
	const float value = values[action].item().toFloat();

	values[action] += (reward - value) * alpha;
}
