/*
 * bandit.h
 *
 *  Created on: Dec 8, 2020
 *      Author: zf
 */

#ifndef CH2_BANDIT_H_
#define CH2_BANDIT_H_

#include <torch/torch.h>
#include <vector>

class Bandit {
private:
	const int armNum;
	torch::Tensor rewards;
	int bestAction;

public:
	Bandit(int num);
	~Bandit() = default;

	float takeAction(int action);
	const int getBestAction();

	void updateRewards(float scale);
};


#endif /* CH2_BANDIT_H_ */
