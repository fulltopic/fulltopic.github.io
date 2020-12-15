/*
 * orig4_5.cpp
 *
 *  Created on: Dec 14, 2020
 *      Author: zf
 */


#include <vector>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string>
//#include <torch/torch.h>

namespace {
const int MaxCar = 20;
const int MaxParkCar = 10;
const int MaxMove = 5;
const int MinMove = -5;
const float Gamma = 0.9;

const float RentReward = 10;
const float MoveReward = -2;
const float ParkReward = -4;
const int FreeMove = 1;

const float theta = 0.01;

std::vector<std::vector<std::vector<float>>> vs(2, std::vector<std::vector<float>>(MaxCar + 1, std::vector<float>(MaxCar + 1, 0.0f)));
std::vector<std::vector<float>> a(MaxCar + 1, std::vector<float>(MaxCar + 1, 0.0f));

float getPoisson(float k, float lamda) {
	double facK = 1;
	for (int i = k; i > 0; i --) {
		facK *= i;
	}

//	std::cout << pow(lamda, k) << std::endl;
//	std::cout << exp(-1 * lamda) << std::endl;
//	std::cout << facK << std::endl;
	double r = pow(lamda, k) * exp(-1 * lamda) / facK;

	return r;
}

float getL1Rent(float k) {
	static const float lamda = 3;
	static std::map<int, float> pvs;

	if (pvs.find((int)k) == pvs.end()) {
		auto p = getPoisson(k, lamda);
		pvs[(int)k] = p;
		return p;
	} else {
		return pvs[(int)k];
	}
}

float getL2Rent(float k) {
	static const float lamda = 4;
	static std::map<int, float> pvs;

	if (pvs.find((int)k) == pvs.end()) {
		auto p = getPoisson(k, lamda);
		pvs[(int)k] = p;
		return p;
	} else {
		return pvs[(int)k];
	}}

float getL1Return(float k) {
	static const float lamda = 3;
	static std::map<int, float> pvs;

	if (pvs.find((int)k) == pvs.end()) {
		auto p = getPoisson(k, lamda);
		pvs[(int)k] = p;
		return p;
	} else {
		return pvs[(int)k];
	}
}

float getL2Return(float k) {
	static const float lamda = 2;
	static std::map<int, float> pvs;

	if (pvs.find((int)k) == pvs.end()) {
		auto p = getPoisson(k, lamda);
		pvs[(int)k] = p;
		return p;
	} else {
		return pvs[(int)k];
	}
}

float getExpReward(int num1, int num2) {
	float reward = 0;

	for (int i = 0; i <= num1; i ++) {
		reward += getL1Rent(i) * i;
	}
	for (int i = 0; i <= num2; i ++) {
		reward += getL2Rent(i) * i;
	}

	return reward * RentReward;
}

float getNextFactor(int num1, int num2, int action, const int vIndex) {
	float reward = getExpReward(num1, num2);
//	reward += abs(action) * MoveReward; //Orig
	reward += std::max(abs(action) - 1, 0) * MoveReward; //4.5

	num1 = (int)std::min(std::max(num1 - action, 0), MaxCar);
	num2 = (int)std::min(std::max(num2 + action, 0), MaxCar);

	//4.5
	if (num1 > MaxParkCar) {
		reward += ParkReward;
	}
	if (num2 > MaxParkCar) {
		reward += ParkReward;
	}
	//End 4.5

	float sumV = 0;

	//TODO: Maybe 2 * MaxCar
	for (int return1 = 0; return1 < MaxCar; return1 ++) {
		for (int return2 = 0; return2 < MaxCar; return2 ++) {
			float p1 = getL1Return(return1);
			float p2 = getL2Return(return2);
			float p = p1 * p2;

			int nextNum1 = (int)std::min(std::max(num1 + return1, 0), MaxCar);
			int nextNum2 = (int)std::min(std::max(num2 + return2, 0), MaxCar);

			sumV += p * (reward + Gamma * vs[vIndex][nextNum1][nextNum2]);
		}
	}

	return sumV;
}

void printVs(const std::vector<std::vector<float>>& data, const std::string comment) {
	std::cout << comment << std::endl;
	for (int i = data.size() - 1; i >= 0; i --) {
//	for (int i = 0; i < data.size(); i ++) {
		for (int j = 0; j < data[0].size(); j ++) {
			std::cout << data[i][j] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void valueEval() {
	float delta = 0;
	int index = 0;
	int round = 0;

	do {
		std::cout << "Value iteration round " << round << std::endl;
		round ++;

		index = 1- index;
		int backIndex = 1 - index;
		delta = 0;

		for (int i = 0; i < vs[0].size(); i ++) {
			for (int j = 0; j < vs[0][0].size(); j ++) {
				vs[index][i][j] = getNextFactor(i, j, a[i][j], backIndex);
				delta = std::max((float)abs(vs[index][i][j] - vs[backIndex][i][j]), delta);
			}
		}
	} while (delta > theta);

//	printVs(vs[0], "Stable values");
}

void policyImprove() {
	bool isStable = false;
	int round = 0;

	while (!isStable) {
		std::cout << "Policy improve round " << round << std::endl;
		round ++;

		valueEval();

		isStable = true;
		for (int i = 0; i < vs[0].size(); i ++) {
			for (int j = 0; j < vs[0][0].size(); j ++) {
				int origA = (int)a[i][j];
				int maxA = origA;
				float maxV = -100000;

				for (int action = (-1) * j; action <= i; action ++) {
					if (action > MaxMove || action < (-1) * MaxMove) {
						continue;
					}
					auto alterV = getNextFactor(i, j, action, 0);
					if (alterV > maxV) {
						maxV = alterV;
						maxA = action;
					}
				}

				a[i][j] = maxA;
				if (maxA != origA) {
					isStable = false;
				}
			}
		}
		printVs(a, "actions");
	}
}


}

int main() {
	policyImprove();
//	std::cout << getPoisson(20, 4) << std::endl;
}

