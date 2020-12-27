/*
 * shortcut.cpp
 *
 *  Created on: Dec 27, 2020
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
template<int W, int H>
class State {
public:
	int x;
	int y;

	State (const int w, const int h): x(w), y(h) {}

	State& operator= (const State& other) {
		x = other.x;
		y = other.y;
		return *this;
	}

	bool operator< (const State& other) const{
		if (x != other.x) {
			return x < other.x;
		}
		if (y != other.y) {
			return y < other.y;
		}

		return false;
	}
};

template<int W, int H>
State<W, H> operator+ (const State<W, H>& a, const State<W, H>& b) {
	return {
		std::min(std::max(a.x + b.x, 0), W - 1),
		std::min(std::max(a.y + b.y, 0), H - 1)
	};
}
template<int W, int H>
bool operator== (const State<W, H>& a, const State<W, H>& b) {
	return a.x == b.x && a.y == b.y;
}

class Grid {
public:
	static constexpr const int Width = 10;
	static constexpr const int Height = 7;
	static const int ActionSize;

	using GridState = State<Width, Height>;

	static const GridState Start;
	static const GridState Term;

	static const int BlockY;
	static const std::vector<std::vector<int>> BlockX;

	static const std::vector<GridState> ActionDelta;

	int blockIndex = 0;

	Grid(){
		srand(time(NULL));
	}

	void change() {
		blockIndex = 1;
	}
	void reset() {
		blockIndex = 0;
	}

	std::tuple<GridState, float, bool> takeAction(const GridState& state, int action) {
		GridState nextState = state + ActionDelta[action];

		if (nextState == Term) {
			return {Start, 1, true};
		}

		if (nextState.y == BlockY && nextState.x <= BlockX[blockIndex][1] && nextState.x >= BlockX[blockIndex][0]) {
			return {{state.x, state.y}, 0, false};
		}

		return {nextState, 0, false};
	}

};
using GridState = State<Grid::Width, Grid::Height>;
const GridState Grid::Start {0, 3};
const GridState Grid::Term {8, 5};
const int Grid::ActionSize = 4;
const std::vector<GridState> Grid::ActionDelta = {
		{0, -1},
		{0, 1},
		{-1, 0},
		{1, 0},
//		{-1, -1},
//		{1, 1},
//		{-1, 1},
//		{1, -1},
//		{0, 0}
};
const int Grid::BlockY = 2;
const std::vector<std::vector<int>> Grid::BlockX = {{0, 8}, {1, 9}};


class GreedyPolicy {
public:
	const float epsilon;
	const int actionSize;

	GreedyPolicy(float e, int size): epsilon(e), actionSize(size) {
		srand(time(NULL));
	}

	int getAction(torch::Tensor& qsValues) {
		static const torch::Tensor probTensor(torch::ones({1}) * epsilon);
		static const torch::Tensor countTensor = torch::ones({1});

		static const torch::Tensor tieTensor (torch::ones({1}) * 0.5);

		torch::Tensor greedyOutTensor = torch::binomial(countTensor, probTensor);
		int greedyOutput = greedyOutTensor.item().toInt();

		if (greedyOutput == 1) {
			return (int)(torch::rand({1}).item().toFloat() * actionSize);

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
		return qsValues.argmax().item().toInt();
	}
};

using QType = std::map<GridState, torch::Tensor>;
void initQs(QType& qs) {
	for (int i = 0; i < Grid::Width; i ++) {
		for (int j = 0; j < Grid::Height; j ++) {
			qs[{i,j}] = torch::zeros({Grid::ActionSize});
		}
	}
}

const float epsilon = 0.2;
const float gamma = 0.95;
const float alpha = 0.1;
const int planStep = 50;
const float k = 0.001;

std::vector<float> trainRewards;
std::vector<float> trainPlusRewards;

struct ModelKType {
	GridState s;
	int a;

	bool operator< (const ModelKType& other) const{
		if (s == other.s) {
			return a < other.a;
		} else {
			return s < other.s;
		}
	}

	bool operator== (const ModelKType& other) const {
		return (s == other.s) && (a == other.a);
	}
};

struct ModelVType {
public:
	GridState s;
	float r;

	ModelVType(const GridState ss, const float rr): s(ss), r(rr) {}
	ModelVType():s({0, 0}), r(0) {};


	ModelVType& operator= (const ModelVType&& other) {
		s = other.s;
		r = other.r;

		return *this;
	}
};
using ModelType = std::map<ModelKType, ModelVType>;

void train(const int stepNum, const int changeStep) {
	QType qs;
	QType taos;
	initQs(qs);
	ModelType model;


	Grid env;
	GreedyPolicy policy(epsilon, Grid::ActionSize);

	float sumReward = 0;
	GridState s = Grid::Start;
	GridState nextState = s;
	float reward = 0;
	bool isTerm = false;
	for (int i = 0; i < stepNum; i ++) {
		if (i == changeStep) {
			env.change();
			s = Grid::Start;
			std::cout << "Env changed " << std::endl;
		}
		int a = policy.getAction(qs[s]);
//		std::cout << "S = " << s.x << ", " << s.y << ": " << a << std::endl;
		std::tie(nextState, reward, isTerm) = env.takeAction(s, a);
		qs[s][a] += alpha * (reward + gamma * qs[nextState].max() - qs[s][a]);
		model[{s, a}] = {nextState, reward};
		sumReward += reward;
		trainRewards.push_back(sumReward);

		s = nextState;

		for (int j = 0; j < planStep; j ++) {
			int index = (int)(torch::rand({1}).item().toFloat() * model.size());
			auto it = model.begin();
			std::advance(it, index);

			auto& key = it->first;
			auto& value = it->second;

			qs[key.s][key.a] += alpha * (value.r + gamma * qs[value.s].max() - qs[key.s][key.a]);
		}
	}
	std::cout << "Reward sum: " << sumReward << std::endl;
}


void trainPlus(const int stepNum, const int changeStep) {
	QType qs;
	QType taos;
	initQs(qs);
	ModelType model;

	QType lastVisits;
	initQs(lastVisits);

	Grid env;
	GreedyPolicy policy(epsilon, Grid::ActionSize);

	env.reset();
	float sumReward = 0;
	GridState s = Grid::Start;
	GridState nextState = s;
	float reward = 0;
	bool isTerm = false;
	for (int i = 0; i < stepNum; i ++) {
		if (i == changeStep) {
			env.change();
			s = Grid::Start;
			std::cout << "Env changed " << std::endl;
		}
//		std::cout << "S = " << s.x << ", " << s.y << std::endl;
		torch::Tensor tao = (i * torch::ones({Grid::ActionSize}) - lastVisits[s]).sqrt() * k;
		auto q = qs[s] + tao;
		int a = policy.getAction(q);
		std::tie(nextState, reward, isTerm) = env.takeAction(s, a);

		qs[s][a] += alpha * (reward + gamma * qs[nextState].max() - qs[s][a]);
		model[{s, a}] = {nextState, reward};
		lastVisits[s][a] = i;
		sumReward += reward;
		trainPlusRewards.push_back(sumReward);
		s = nextState;

		for (int j = 0; j < planStep; j ++) {
			int index = (int)(torch::rand({1}).item().toFloat() * model.size());
			auto it = model.begin();
			std::advance(it, index);

			auto& key = it->first;
			auto& value = it->second;

			qs[key.s][key.a] += alpha * (value.r + gamma * qs[value.s].max() - qs[key.s][key.a]);
		}
	}
	std::cout << "Reward sum: " << sumReward << std::endl;

}

void plotReward() {
	matplotlibcpp::clf();
	matplotlibcpp::plot(trainRewards, "r--");
	matplotlibcpp::plot(trainPlusRewards, "g");
	matplotlibcpp::pause(5);
	matplotlibcpp::save("test.png");
}

}

int main() {
	int stepNum = 6000;
	int changeStep = 3000;
	train(stepNum, changeStep);
	trainPlus(stepNum, changeStep);
	plotReward();
//	train(2000, 2000);
//	trainPlus(2000, 2000);
}
