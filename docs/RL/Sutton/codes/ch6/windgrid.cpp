/*
 * windgrid.cpp
 *
 *  Created on: Dec 21, 2020
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

#include <torch/torch.h>

namespace {
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

State operator+ (const State& a, const State& b) {
	return {a.x + b.x, a.y + b.y};
}
bool operator== (const State& a, const State& b) {
	return a.x == b.x && a.y == b.y;
}

class Grid {
public:
	static const int Width = 10;
	static const int Height = 7;
	static const int ActionSize = 9;

	static const State Start;
	static const State Term;

	static const std::vector<State> ActionDelta;
	static const std::vector<int> Wind;

	Grid() {
		srand(time(NULL));
	}

	int getWind(const State& state) {
		return Wind[state.x] + (torch::rand({1}).item().toInt() % 3) - 1;
	}

	std::tuple<State, float, bool> takeAction(const State& state, int action) {
		State nextState = state + ActionDelta[action];
		nextState.y += Wind[nextState.x];

//		if (nextState.x >= Width || nextState.x < 0 || nextState.y >= Height || nextState.y < 0) {
//			return {nextState, std::numeric_limits<float>::min(), true};
//		}
		if (nextState == Term) {
			return {nextState, 0, true};
		}

		nextState.x = std::min(Width - 1, std::max(0, nextState.x));
		nextState.y = std::min(Height - 1, std::max(0, nextState.y));
		return {nextState, -1, false};
	}

	std::tuple<State, float, bool> takeStochAction(const State& state, int action) {
		State nextState = state + ActionDelta[action];
		nextState.y += getWind(nextState);

//		if (nextState.x >= Width || nextState.x < 0 || nextState.y >= Height || nextState.y < 0) {
//			return {nextState, std::numeric_limits<float>::min(), true};
//		}
		if (nextState == Term) {
			return {nextState, 0, true};
		}

		nextState.x = std::min(Width - 1, std::max(0, nextState.x));
		nextState.y = std::min(Height - 1, std::max(0, nextState.y));
		return {nextState, -1, false};
	}
};
const State Grid::Start {0, 3};
const State Grid::Term {7, 3};
const std::vector<State> Grid::ActionDelta = {
		{0, -1},
		{0, 1},
		{-1, 0},
		{1, 0},
		{-1, -1},
		{1, 1},
		{-1, 1},
		{1, -1},
		{0, 0}
};
const std::vector<int> Grid::Wind = {
		0, 0, 0, 1, 1, 1, 2, 2, 1, 0
};

using QType = std::map<State, torch::Tensor>;
class GreedyPolicy {
public:
	const float epsilon;
	const int actionSize;

	GreedyPolicy(float e, int size): epsilon(e), actionSize(size) {
		srand(time(NULL));
	}

	int getAction(QType& qs, State& s) {
		static const torch::Tensor probTensor(torch::ones({1}) * epsilon);
		static const torch::Tensor countTensor = torch::ones({1});
		torch::Tensor greedyOutTensor = torch::binomial(countTensor, probTensor);
		int greedyOutput = greedyOutTensor.item().toInt();

		if (greedyOutput == 1) {
			return rand() % actionSize;
		} else {
			return qs[s].argmax(0).item().toInt();
		}
	}

	int getRunAction(QType& qs, State& s) {
		return qs[s].argmax(0).item().toInt();
	}
};

void initQs(QType& qs) {
	for (int i = 0; i < Grid::Width; i ++) {
		for (int j = 0; j < Grid::Height; j ++) {
			qs[{i,j}] = torch::zeros({Grid::ActionSize});
		}
	}
}


void test(QType& qs) {
	Grid env;
	State state = Grid::Start;
	bool isTerm = false;
	GreedyPolicy policy(0, Grid::ActionSize);
	float reward;
	while(!isTerm) {
		auto a = policy.getRunAction(qs, state);
		std::cout << "RunState " << state.x << ", " << state.y << ": " << Grid::ActionDelta[a].x << ", " << Grid::ActionDelta[a].y << std::endl;
//		std::tie(state, reward, isTerm) = env.takeAction(state, a);
		std::tie(state, reward, isTerm) = env.takeStochAction(state, a);
	}
}

void train() {
	Grid env;

	QType qs;
	initQs(qs);
	GreedyPolicy policy(0.1, Grid::ActionSize);

	const int epochNum = 4000;
	const float gamma = 1;
	const float alpha = 0.1;
	for (int i = 0; i < epochNum; i ++) {
//		std::cout << "Episode " << i << std::endl;
		bool isTerm = false;
		State state = Grid::Start;
		State nextState = state;
		int action = policy.getAction(qs, state);
		int nextAction;
		float reward = 0;
		while (!isTerm) {
//			std::cout << "State " << state.x << ", " << state.y << ": " << Grid::ActionDelta[action].x << ", " << Grid::ActionDelta[action].y << std::endl;
//			std::tie(nextState, reward, isTerm) = env.takeAction(state, action);
			std::tie(nextState, reward, isTerm) = env.takeStochAction(state, action);
			if (isTerm) {
				break;
			}
			nextAction = policy.getAction(qs, nextState);
			qs[state][action] += alpha * (reward + gamma * qs[nextState][nextAction] - qs[state][action]);

			state = nextState;
			action = nextAction;
		}
	}

	test(qs);
}
}

int main() {
	train();
}
