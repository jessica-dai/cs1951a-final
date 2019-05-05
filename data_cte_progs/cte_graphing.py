import numpy as np
from cte_process import get_reg, get_u_type, get_conditionals # TODO
import matplotlib.pyplot as plt 
from matplotlib.colors import ColorConverter
import seaborn as sns

sns.set_style("white")
colors = ['#db7093', '#800080', '#6495ed', '#9acd32']

# colors = ['#ffb6c1', '#db7093', '#ba55d3', '#800080', \
#             '#695ed', '#00ced1', '#3cb371', '#32cd32', \
#             '#008000', ]

def conditional_data():
    
    raw_urb = [232, 489, 309, 482]
    raw_reg = [271, 318, 467, 456]

    urb_1std = [56, 91, 50, 53]
    reg_1std = [64, 54, 61, 71]

    urb_2std = [8, 21, 7, 5]
    reg_2std = [17, 9, 12, 3]

    w_u1 = [float(c)/float(r) for c, r in zip(urb_1std, raw_urb)]
    r_u1 = [float(c)/float(r) for c, r in zip(reg_1std, raw_reg)]
    w_u2 = [float(c)/float(r) for c, r in zip(urb_2std, raw_urb)]
    r_u2 = [float(c)/float(r) for c, r in zip(reg_2std, raw_reg)]

    return w_u1, w_u2, r_u1, r_u2

def graph_cond(data, labels, n, tp):
    x = np.arange(4)
    plt.bar(x, data, color=colors)
    plt.xticks(x, labels)
    plt.ylim(0, 0.30)
    plt.gca().set(title='Conditional probability of score ' + str(n) + 'standard deviations above the mean given ' + tp)
    plt.show()

def stacked_cond(data1, data2, labels, tp):
    x = np.arange(4)
    # width = 0.35
    diff = np.array(data1) - np.array(data2)
    p1 = plt.bar(x, data1)
    p2 = plt.bar(x, data2, bottom=diff)
    plt.xticks(x, labels)
    plt.ylim(0, 0.40)
    plt.gca().set(title='Conditional probability of program quality score given ' + tp)
    plt.legend((p1[0], p2[0]), ('1 standard deviation above the mean', '2 standard deviations above the mean'))
    plt.show()

def graph(categories, labels, key):

    # kwargs = dict(alpha=0.5, bins=100)
    kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
    # kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

    for i in range(len(categories)):
        # sns.distplot(categories[i], color=ColorConverter.to_rgb(colors[i]), label=labels[i], **kwargs)
        plt.hist(categories[i], **kwargs, color=ColorConverter.to_rgb(colors[i]), label=labels[i])

    plt.gca().set(title='Program quality by ' + key + ', normalized by population size', ylabel='Probability (normalized frequency)')
    # plt.xlim(50,75)
    plt.legend()

    plt.show()



def main():

    # labels
    r_labels = ['northeast', 'southeast', 'central', 'west']
    t_labels = ['city', 'suburban', 'town', 'rural']

    # data
    r_data = []
    type_data = []
    for i in range(1, 5):
        r_data.append(get_reg(i))
        type_data.append(get_u_type(i))
    
    # graph
    # graph(r_data, r_labels, 'region')
    # graph(type_data, t_labels, 'community type')

    w_u1, w_u2, r_u1, r_u2 = conditional_data()
    # graph_cond(w_u1, t_labels, 1, 'community type')
    # graph_cond(w_u2, t_labels, 2, 'community type')
    # graph_cond(r_u1, t_labels, 1, 'region')
    # graph_cond(r_u2, t_labels, 2, 'region')

    stacked_cond(w_u1, w_u2, t_labels, 'community type')
    stacked_cond(r_u1, r_u2, r_labels, 'region')

if __name__ == "__main__":
    main() 