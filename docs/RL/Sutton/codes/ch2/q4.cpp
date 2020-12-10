/*
 * q4.cpp
 *
 *  Created on: Dec 9, 2020
 *      Author: zf
 */




/*
 * q1.cpp
 *
 *  Created on: Dec 8, 2020
 *      Author: zf
 */


#include <matplotlibcpp.h>
#include <torch/torch.h>
#include "bandit.h"
#include "ch2model.h"
#include "ch2modelgreedy.h"
#include "ch2modelconst.h"
#include <vector>
#include <map>
#include <string>
#include <memory>

namespace {
template <typename T>
void plotQ1(std::vector<std::vector<T>>& rewards, const std::string fileName) {
	const static std::vector<std::string> colors { "r--", "g", "b"};

	matplotlibcpp::clf();
	matplotlibcpp::plot(rewards[0], colors[0]);
	matplotlibcpp::plot(rewards[1], colors[1]);
//	matplotlibcpp::plot(rewards[2], colors[2]);
	matplotlibcpp::pause(5);
	matplotlibcpp::save(fileName);
}

const int sampleNum = 1000;
const int bsNum = 2;

std::vector<std::vector<float>> globalRewards(bsNum, std::vector<float>(sampleNum, 0.0f));
std::vector<std::vector<int>> globalHits(bsNum, std::vector<int>(sampleNum, 0));

void test() {
	const int armNum = 10;

	std::vector<Bandit> bs;
	std::vector<std::unique_ptr<Ch2Model>> ms;
	const float scale = 0.8;
	const float epsilon = 0.1;
	const float alpha = 0.1;
	for (int i = 0; i < bsNum; i ++) {
		bs.push_back(Bandit(armNum));
	}
	ms.push_back(std::make_unique<Ch2GreedyModel>(armNum, epsilon));
	ms.push_back(std::make_unique<Ch2ConstModel>(armNum, alpha, epsilon));

	std::vector<std::vector<float>> totalRewards(bsNum, std::vector<float>(sampleNum, 0.0f));
	std::vector<std::vector<int>> hits(bsNum, std::vector<int>(sampleNum, 0));
	for (int i = 0; i < sampleNum; i ++) {
		for (int j = 0; j < bsNum; j ++) {
			int action = ms[j]->getAction();
			float reward = bs[j].takeAction(action);

			if (i == 0) {
				totalRewards[j][i] = reward;
				if (action == bs[j].getBestAction()) {
					hits[j][i] = 1;
				}
			} else {
				totalRewards[j][i] = totalRewards[j][i - 1] + (reward - totalRewards[j][i]) / (i + 1);
				if (action == bs[j].getBestAction()) {
					hits[j][i] = hits[j][i - 1] + 1;
//					std::cout << "j=" << es[j] << " hit " << hits[j][i] << std::endl;
				} else {
//					std::cout << "j=" << es[j] << ": " << action << " != " << bs[j].getBestAction() << std::endl;
					hits[j][i] = hits[j][i - 1];
				}
			}

			ms[j]->update(action, reward);
			bs[j].updateRewards(scale);
		}
	}

	for (int i = 0; i < totalRewards.size(); i ++) {
		for (int j = 0; j < totalRewards[i].size(); j ++) {
			globalRewards[i][j] += totalRewards[i][j];
			globalHits[i][j] += hits[i][j];
		}
	}

}

void testEpoch() {
	const int epochNum = 8;
	for (int i = 0; i < epochNum; i ++) {
		test();
	}

	for (int i = 0; i < globalRewards.size(); i ++) {
		for (int j = 0; j < globalRewards[i].size(); j ++) {
			globalRewards[i][j] /= epochNum;
			globalHits[i][j] /= epochNum;
		}
	}

	plotQ1(globalRewards, "reward.png");
	plotQ1(globalHits, "hits.png");
}
}

int main() {
//	test();
	testEpoch();
}
