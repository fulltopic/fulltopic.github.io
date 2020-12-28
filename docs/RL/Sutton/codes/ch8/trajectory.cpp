/*
 * trajectory.cpp
 *
 *  Created on: Dec 28, 2020
 *      Author: zf
 */



#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
#include <random>
#include <limits>
#include <iterator>

#include <matplotlibcpp.h>

#include <torch/torch.h>

namespace {
const int b = 1;
constexpr int ANum = 2;
const int SNum = 10000;
const float termProb = 0.1;
const float Epsilon = 0.1;
const float gamma = 1;
const float alpha = 0.1;

struct NextType {
	int s;
	float r;
};

using EnvType = std::vector<std::vector<std::vector<NextType>>>;
class Env {
public:
	const int sNum;
	const int termS;
	EnvType trans;

	Env(int num): sNum(num), termS(num) {}

	void buildEnv() {
		for (int i = 0; i < sNum; i ++) {
			std::vector<std::vector<NextType>> as;
			for (int j = 0; j < ANum; j ++) {
				std::vector<NextType> ss;
				for (int k = 0; k < b; k ++) {
					int s = (int)(torch::rand({1}).item().toFloat() * (sNum - 1));
					float r = torch::randn({1}).item().toFloat();
					ss.push_back({s, r});
				}
				float r = torch::randn({1}).item().toFloat();
				ss.push_back({termS, r}); //terminal

				as.push_back(ss);
			}
			trans.push_back(as);
		}
	}

	std::tuple<int, float, bool> takeAction(int s, int a) {
		static const torch::Tensor probTensor(torch::ones({1}) * termProb);
		static const torch::Tensor countTensor = torch::ones({1});

		int greedyOutput = torch::binomial(countTensor, probTensor).item().toInt();
		if (greedyOutput == 1) {
			return {trans[s][a][b].s, trans[s][a][b].r, true};
		} else {
			int next = (int)(torch::rand({1}).item().toFloat() * b);
			return {trans[s][a][next].s,  trans[s][a][next].r, false};
		}
	}
};

class RandomPolicy {
public:
	int getAction() {
		static const torch::Tensor countTensor = torch::ones({1});
		static const torch::Tensor probTensor (torch::ones({1}) * 0.5);

		torch::Tensor greedyOutTensor = torch::binomial(countTensor, probTensor);
		int greedyOutput = greedyOutTensor.item().toInt();

		if (greedyOutput == 1) {
			return 1;
		} else {
			return 0;
		}
	}
};

class GreedyPolicy {
public:
	const int actionSize;
	const torch::Tensor probTensor;

	GreedyPolicy(float e, int size): actionSize(size), probTensor(torch::ones({1}) * e) {
		srand(time(NULL));
	}

	int getAction(torch::Tensor& qsValues) {
		static const torch::Tensor countTensor = torch::ones({1});
		static const torch::Tensor tieTensor (torch::ones({1}) * 0.5);

		torch::Tensor greedyOutTensor = torch::binomial(countTensor, probTensor);
		int greedyOutput = greedyOutTensor.item().toInt();

		if (greedyOutput == 1) {
			int index = (int)(torch::rand({1}).item().toFloat() * actionSize);
			return index;
		} else {
			auto data = qsValues.data_ptr<float>();
			auto maxValue = -10000;
			int index = 0;
			for (int i = 0; i < qsValues.numel(); i ++) {
				if (data[i] > maxValue) {
					maxValue = data[i];
					index = i;
				} else if (data[i] == maxValue) {
					torch::Tensor greedyOutTensor = torch::binomial(countTensor, tieTensor);
					int greedyOutput = greedyOutTensor.item().toInt();
					if (greedyOutput == 1) {
						index = i;
					}
				}
			}

			return index;
		}
	}

	int getRunAction(torch::Tensor& qsValues) {
		return qsValues.argmax(0).item().toInt();
	}
};


using QType = std::map<int, torch::Tensor>;
void initQs(QType& qs) {
	for (int i = 0; i <= SNum; i ++) {
		qs[i] = torch::zeros({ANum});
	}
}

float computeS(Env& env, int ss, QType& qs) {
	GreedyPolicy policy(0, ANum);

	std::vector<float> rewards;
	int s = ss;
	int nextS = s;
	bool isTerm = false;
	float reward = 0;

	while (!isTerm) {
		int a = policy.getRunAction(qs[s]);
		std::tie(nextS, reward, isTerm) = env.takeAction(s, a);
		rewards.push_back(reward);

		s = nextS;
	}

	float v = 0;
	for (int i = rewards.size() - 1; i >= 0; i --) {
		v = v * gamma + rewards[i];
	}

	return v;
}

const int StepNum = 40000;
const int StartState = SNum / 2;
const int epochNum = 10;
const int testStep = 1000;


std::vector<float> trainRandom(Env& env) {
	QType qs;
	initQs(qs);
	RandomPolicy policy;

	std::vector<float> returns;


	static const torch::Tensor probTensor(torch::ones({1}) * 0.5);
	static const torch::Tensor countTensor = torch::ones({1});

	for (int i = 0; i < StepNum; i ++) {
		int sIndex = (int)(torch::rand({1}).item().toFloat() * (SNum - 1));
		int probOutput = torch::binomial(countTensor, probTensor).item().toInt();

		int a = 0;
		if (probOutput == 1) {
			a = 1;
		}

		auto updateV = qs[sIndex][a] * 0;
		const auto& next = env.trans[sIndex][a];
		for (int j = 0; j < b; j ++) {
			auto nextV = (1 - termProb) / b * (next[j].r + gamma * qs[next[j].s].max());
			updateV += nextV;
		}
		updateV += termProb * next[b].r;
		qs[sIndex][a] = updateV;

		if ((i % testStep) == 0) {
			returns.push_back(computeS(env, StartState, qs));
		}
	}

	return returns;
}

std::vector<float>  trainOnPolicy(Env& env) {
	QType qs;
	initQs(qs);
	GreedyPolicy policy(Epsilon, ANum);

	std::vector<float> returns;

	int s = StartState;
	int nextS = s;
	int a = 0;
	float reward = 0;
	bool isTerm = true;

	for (int i = 0; i < StepNum; i ++) {
		if (isTerm) {
			s = StartState;
		}

		a = policy.getAction(qs[s]);
		std::tie(nextS, reward, isTerm) = env.takeAction(s, a);

		if (isTerm) {
			qs[s][a] += alpha * (reward - qs[s][a]);
		} else {
			qs[s][a] += alpha * (reward + gamma * qs[nextS].max() - qs[s][a]);
		}
		s = nextS;

		if ((i % testStep) == 0) {
			returns.push_back(computeS(env, StartState, qs));
		}
	}

	return returns;
}
}

int main() {
	Env env(SNum);
	env.buildEnv();

	std::vector<std::vector<float>> randomReturns;
	std::vector<std::vector<float>> policyReturns;
	for (int i = 0; i < epochNum; i ++) {
		randomReturns.push_back(trainRandom(env));
		policyReturns.push_back(trainOnPolicy(env));
	}

	std::vector<float> randomReturn(randomReturns[0].size(), 0);
	std::vector<float> policyReturn(randomReturns[0].size(), 0);
	for (int i = 0; i < randomReturns[0].size(); i ++) {
		for (int j = 0; j < randomReturns.size(); j ++) {
			randomReturn[i] += randomReturns[j][i];
			policyReturn[i] += policyReturns[j][i];
		}

		randomReturn[i] /= randomReturns.size();
		policyReturn[i] /= policyReturns.size();
	}

	matplotlibcpp::clf();
	matplotlibcpp::plot(randomReturn, "r--");
	matplotlibcpp::plot(policyReturn, "g");
	matplotlibcpp::pause(5);
	matplotlibcpp::save("plan.png");
}


