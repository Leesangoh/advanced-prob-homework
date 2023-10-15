import numpy as np
import matplotlib.pyplot as plt

poisson_pmf = lambda k, l: (l**k / np.math.factorial(k)) * np.exp(-l)
binomial_pmf = lambda k, n, p: np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k)) * (p**k) * ((1 - p)**(n - k))

# By letting l = n * p, let us show that binomial pmf can be approximted by poisson pmf

# Let n = 100, p = 0.1
n_list = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
p_list = [0.16, 0.32, 0.48, 0.64, 0.8, 0.96]

x_lim_list = [(0,30), (0, 60), (0, 80), (0, 100), (0, 120), (0, 120)]

kl_dict = {}

color_per_n_list = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'pink', 'orange', 'gray', 'brown']


for p in p_list:
    print("p = ", p)
    plt.figure(figsize=(15, 10))
    for i, n in enumerate(n_list):

        l = n * p

        # plot the binomial pmf
        binomial_pmf_list = []
        for k in range(0, n + 1):
            binomial_pmf_list.append(binomial_pmf(k, n, p))
            poisson_pmf_list = [poisson_pmf(k, l) for k in range(0, n + 1)]

        # calculate the kullback-leibler divergence
        kl_divergence = 0
        for k in range(0, n + 1):
            kl_divergence += binomial_pmf_list[k] * np.log(binomial_pmf_list[k] / poisson_pmf_list[k])

        kl_sci = "{:.2e}".format(kl_divergence)

        try:
            a, b = kl_sci.split('e-')
            print_str = '$' + a + ' \cdot 10^{-' + str(int(b)) + '}' + '$'
        except:
            a, b = kl_sci.split('e+')
            print_str = '$' + a + ' \cdot 10^{' + str(int(b)) + '}' + '$'

        

        kl_dict[(n, p)] = print_str

        plt.plot(range(0, n + 1), binomial_pmf_list, label='binomial pmf', marker='o', color = color_per_n_list[i], linestyle = 'dashed', markersize=5)
        plt.plot(range(0, n + 1), poisson_pmf_list, label='poisson pmf', marker='o', color = color_per_n_list[i], markersize=5)
    
    # set legend from its color. 'red': p=0.1, 'blue': p=0.2, ..., 'brown': p=0.9
    # set text size to 20
    plt.legend(loc='upper right', labels=['n = ' + str(n) + ', ' + distr_type  for n in n_list for distr_type in ['Binomial', 'Poisson']], prop={'size': 15})
    plt.title(f'Binomial pmf and Poisson pmf with p = {p}', fontsize=30)
    plt.xlabel('k', fontsize=20)
    plt.xlim(x_lim_list[p_list.index(p)])

    # set x limit margin
    plt.xlim(x_lim_list[p_list.index(p)][0] - 0.5, x_lim_list[p_list.index(p)][1] + 0.5)

    # set x axis fontsize 20
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('probability', fontsize=25)
    

    # tighten
    plt.tight_layout()
    plt.savefig(f'binomial_pmf_poisson_pmf_p_{p}.pdf', dpi = 300)


# print kl_dict in latex table format

print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}")
print("\\hline")
print("p & n = 1 & n = 4 & n = 9 & n = 16 & n = 25 & n = 36 & n = 49 & n = 64 & n = 81 & n = 100 \\\\ \\hline")
for p in p_list:
    print(f"{p} & {kl_dict[(1, p)]} & {kl_dict[(4, p)]} & {kl_dict[(9, p)]} & {kl_dict[(16, p)]} & {kl_dict[(25, p)]} & {kl_dict[(36, p)]} & {kl_dict[(49, p)]} & {kl_dict[(64, p)]} & {kl_dict[(81, p)]} & {kl_dict[(100, p)]} \\\\ \\hline")
print("\\end{tabular}")
print("\\caption{Kullback-Leibler divergence between binomial pmf and poisson pmf}")
print("\\label{tab:my_label}")
print("\\end{table}")
