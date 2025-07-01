import numpy as np
import matplotlib.pyplot as plt
from trainer import Trainer
from res_unet import Res_Unet
import torch

class HP_optimizer():
    def __init__(self, NP=60, select_ratio=0.8, L=4, G=20, Pc=0.8, Pm=0.05, time_out=0.1, requirement_accuracy=0.7):
        '''
        初始化超参数优化器
        :param NP: 种群数目
        :param select_ratio: 选择比例
        :param L: 染色体长度
        :param G: 进化代数
        :param Pc: 交叉概率
        :param Pm: 变异概率
        :param requirement_fitness: 要求适应度值
        '''
        self.NP = NP
        self.select_ratio = select_ratio
        self.L = L
        self.G = G
        self.Pc = Pc
        self.Pm = Pm
        self.time_out = time_out
        self.requirement_accuracy = requirement_accuracy
        self.requirement_fitness = float(self.fitness_func(verification_time=np.array([time_out]), accuracy=np.array([requirement_accuracy])))

        # 初始化种群
        self.population = np.zeros((NP, L))
        for i in range(NP):
            self.population[i, 0] = np.random.uniform(0.000001, 0.001)  # lr
            self.population[i, 1] = np.random.uniform(0.00001, 0.01)  # warmup_proption
            self.population[i, 2] = np.random.uniform(0.000001, 0.001)    # weight_decay
            self.population[i, 3] = np.random.randint(16, 128)  # batch_size

    # 适应度函数
    def fitness_func(self, verification_time, accuracy):
        '''
        适应度函数
        :param verification_time: 验证时间
        :param accuracy: 准确率
        :return: 适应度值
        '''
        NP = verification_time.shape[0]  # 种群数目
        fitness = []
        for i in range(NP):
            if self.time_out > verification_time[i]:
                fitness.append(accuracy[i] + self.auxiliary_func(verification_time[i], accuracy[i]))
            else:
                fitness.append(accuracy[i] - self.auxiliary_func(verification_time[i], accuracy[i]))

        return np.array(fitness)

    # 辅助函数
    def auxiliary_func(self, verification_time, accuracy):
        '''
        辅助函数
        :param verification_time: 验证时间
        :param accuracy: 准确率
        :return: 辅助函数值
        '''
        Tor = 1 - self.requirement_accuracy  # tolerance
        tor_range = 1 - Tor  # tolerance range
        Zp = accuracy - tor_range  # intercept
        if accuracy > tor_range:
            if verification_time <= self.time_out:
                return -(Tor + Zp) * verification_time + Tor + Zp
            else:
                return -(Tor / (3 - accuracy)) * (verification_time - 1)
        else:
            if verification_time <= self.time_out:
                return -(tor_range - accuracy) * verification_time + (Tor / 4)
            else:
                return -(1 - accuracy) * (verification_time - 1)

    def get_best_hyperparameters(self):
        '''
        获取最优超参数

        :return: 最优超参数
        '''
        average_fitness_list = []  # 平均适应度列表
        best_fitness_list = []  # 最优适应度列表
        best_fitness = -np.inf  # 最优适应度值
        x_best = None  # 最优个体
        count_gen = 0   # 记录优化代数

        # 进化迭代
        for gen in range(self.G):
            # 日志记录
            with open("training_log.txt", "a") as f:
                f.write(f"\nGeneration: {gen + 1}\n")
            # 计算适应度值
            accuracy = np.zeros((self.NP, 1))
            verification_time = np.zeros((self.NP, 1))
            # deep learning
            for i in range(self.NP):
                lr = self.population[i, 0]
                warmup_proption = self.population[i, 1]
                weight_decay = self.population[i, 2]
                batch_size = int(self.population[i, 3])
                # 使用 cuda
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Res_Unet(output_channel=1, pretrained_model_path="pretrained_model/best_finetune_model.pth").to(device)
                trainer = Trainer(
                    model=model,
                    lr=lr,
                    warmup_proportion=warmup_proption,
                    weight_decay=weight_decay,
                    batch_size=batch_size
                )
                acc, veri_t = trainer.train_HP_optim(i)
                accuracy[i] = acc.cpu()
                verification_time[i] = veri_t
            fitness = self.fitness_func(verification_time, accuracy)
            # 记录平均适应度值
            average_fitness = np.mean(fitness)
            average_fitness_list.append(average_fitness)

            index = np.argmax(fitness)  # 最大值索引
            current_x_best = self.population[index].copy()  # 当代最优个体
            current_best_fitness = fitness[index].item()  # 当代最优适应度值
            # 记录当代最优适应度值
            best_fitness_list.append(current_best_fitness)
            # 记录日志：记录每一代平均适应度值、最优适应度值
            with open("training_log.txt", "a") as f:
                f.write(f"\nAverage Fitness: {average_fitness:.4f}\n")
                f.write(f"Best Fitness: {current_best_fitness:.4f}\n")

            # 更新全局最优
            if current_best_fitness > best_fitness:
                x_best = current_x_best
                best_fitness = current_best_fitness
                if best_fitness > self.requirement_fitness:
                    break

            # 归一化
            max_fitness = np.max(fitness)
            min_fitness = np.min(fitness)
            fitness_norm = (fitness - min_fitness) / (max_fitness - min_fitness)

            # 计算选择概率
            P = fitness_norm / np.sum(fitness_norm)
            P = P.flatten()  # 展平为一维

            # 选择（基于轮盘赌）
            selected_indices = np.random.choice(np.arange(self.NP), size=int(self.NP * self.select_ratio), replace=True, p=P)
            selected_population = self.population[selected_indices].copy()
            self.NP = selected_population.shape[0]  # 更新种群数目

            # 交叉
            for i in range(0, self.NP - 1, 2):
                if np.random.rand() < self.Pc:
                    # 随机选择交叉点
                    point = np.random.randint(1, self.L)
                    # 交叉
                    offspring1 = selected_population[i, point:].copy()
                    offspring2 = selected_population[i + 1, point:].copy()
                    selected_population[i, point:], selected_population[i + 1, point:] = offspring2, offspring1

            # 变异
            for i in range(self.NP):
                if np.random.rand() < self.Pm:
                    # 随机选择变异位
                    point = np.random.randint(0, self.L)
                    # 变异
                    if point == 0:  # lr 变异
                        selected_population[i, point] = np.random.uniform(0.0001, 0.01)
                    elif point == 1:  # warmup_propotion 变异
                        selected_population[i, point] = np.random.uniform(0.0001, 0.1)
                    elif point == 2:  # weight_decay 变异
                        selected_population[i, point] = np.random.uniform(0.0001, 0.01)
                    elif point == 3:  # batch_size 变异
                        selected_population[i, point] = np.random.randint(16, 128)

            # 精英策略：将最优个体加入新种群
            reshaped_x_best = x_best.copy().reshape(1, self.L)
            new_population = np.append(selected_population, reshaped_x_best, axis=0)
            self.NP = new_population.shape[0]  # 更新种群数目

            # 更新种群
            self.population = new_population.copy()

            # 更新优化代数
            count_gen += 1

        # 输出结果
        print(f"要求适应度值: {self.requirement_fitness}")
        print(f"最优适应度值: {best_fitness}")
        print("最优超参数:")
        print(f"lr = {x_best[0]}")
        print(f"warmup_propotion = {x_best[1]}")
        print(f"weight_decay = {int(x_best[2])}")
        print(f"batch_size = {int(x_best[3])}")

        # 绘制适应度曲线
        x = np.arange(start=1, stop=count_gen + 1)
        plt.plot(x, best_fitness_list, label='best', markevery=2)
        plt.plot(x, average_fitness_list, label='average', markevery=2)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

        return x_best[0], x_best[1], x_best[2], x_best[3]