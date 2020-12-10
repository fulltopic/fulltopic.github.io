/*
 * bandit.cpp
 *
 *  Created on: Dec 8, 2020
 *      Author: zf
 */

#include "bandit.h"

Bandit::Bandit(int num): armNum(num) {
//	rewards = torch::normal(0.0f, 1.0f, {armNum});
	rewards = torch::randn({armNum});
	bestAction = rewards.argmax(0).item().toInt();
	std::cout << "B" << num << " actions: " << std::endl;
	std::cout << rewards << std::endl;
	std::cout << "best = " << bestAction << std::endl;
	std::cout << "statistic: " << rewards.mean() << std::endl;
	std::cout << "std: " << rewards.std() << std::endl;
}

float Bandit::takeAction(int action) {
	return (torch::rand({1}) + rewards[action]).item().toFloat();
}

const int Bandit::getBestAction() {
	return bestAction;
}

void Bandit::updateRewards(float scale) {
//	torch::Tensor delta = torch::randn({armNum});
//	std::cout << "delta: " << armNum << std::endl << delta << std::endl;
	rewards = rewards.add(torch::randn({armNum}), scale);
}
