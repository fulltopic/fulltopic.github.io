/*
 * ch2modelconst.h
 *
 *  Created on: Dec 9, 2020
 *      Author: zf
 */

#ifndef CH2_CH2MODELCONST_H_
#define CH2_CH2MODELCONST_H_


#include "ch2model.h"

#include <vector>
#include <torch/torch.h>

class Ch2ConstModel: public Ch2Model {
private:
//	const int actionNum;
	const float epsilon;
	const float alpha;
	torch::Tensor values;
//	std::vector<int> counters;
//	long totalCount = 0;

	const torch::Tensor countTensor = torch::ones({1});
	const torch::Tensor probTensor; // = torch::ones({1}) * epsilon;

//	int count1 = 0;
//	int count = 0;
public:
	Ch2ConstModel(int num, float alpha, float e = 0.0f);
	virtual ~Ch2ConstModel() = default;

	virtual int getAction();

	virtual void update(int action, float reward);
};



#endif /* CH2_CH2MODELCONST_H_ */
