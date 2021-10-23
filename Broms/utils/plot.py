import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')


class my_plot():
    def __init__(self, mydata):
        self.data = mydata

    def scatter_plot(self, col_x, col_y, data, *args, group=True):
        '''
        https://betterprogramming.pub/python-user-defined-functions-65f8662e2528
        '''
        if group==True:
            sns.scatterplot(x=col_x, y=col_y, 
                            hue=args,
                            data=data)
            plt.show()
        else:
            sns.scatterplot(x=col_x, y=col_y, 
                            data=data)
            plt.show()

    def count_chart(self, col_x, data, *args, group=True):
        sns.set_theme(style="whitegrid")

        if group==True:
            sns.countplot(x=col_x, hue=args, data=data)
            plt.show()
        else:
            ax = sns.countplot(x=col_x, data=data)
            plt.show()
            

