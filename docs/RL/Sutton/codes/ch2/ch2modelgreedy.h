/*
 * ch2model.h
 *
 *  Created on: Dec 8, 2020
 *      Author: zf
 */

#ifndef CH2_CH2MODELGREEDY_H_
#define CH2_CH2MODELGREEDY_H_

#include "ch2model.h"

#include <vector>
#include <torch/torch.h>

class Ch2GreedyModel: public Ch2Model {
private:
//	const int actionNum;
	float epsilon;
	torch::Tensor values;
	std::vector<int> counters;
//	long totalCount = 0;

	const torch::Tensor countTensor = torch::ones({1});
	const torch::Tensor probTensor; // = torch::ones({1}) * epsilon;

//	int count1 = 0;
//	int count = 0;
public:
	Ch2GreedyModel(int num, float e = 0.0f);
	virtual ~Ch2GreedyModel() = default;

	virtual int getAction();

	virtual void update(int action, float reward);
};



#endif /* CH2_CH2MODELGREEDY_H_ */
