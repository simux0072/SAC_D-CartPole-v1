import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

with plt.style.context('dark_background'):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(20, 10)
    fig.set_dpi(100)

    def animate(i):
        data = pd.read_csv('./database/CartPole-v1.csv')
            
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax5.cla()
        ax6.cla()
        
        ax1.set_title('Score Value')
        ax1.plot(data['Gen'], data['Score'], label='Score', alpha=0.2, color='white')
        ax1.plot(data['Gen'], data['Running Score'], label='Running Score', color='red')
        ax1.legend(loc='upper left')
        
        ax2.set_title('Actor Loss Value')
        ax2.plot(data['Gen'], data['Actor Loss'], label='Actor Loss', alpha=0.2, color='white')
        ax2.plot(data['Gen'], data['Running Actor Loss'], label='Running Actor Loss', color='red')
        ax2.legend(loc='upper left')
        
        ax3.set_title('Critic Loss Value')
        ax3.plot(data['Gen'], data['Critic Loss'], label='Critic Loss', alpha=0.2, color='white')
        ax3.plot(data['Gen'], data['Running Critic Loss'], label='Running Critic Loss', color='red')
        ax3.legend(loc='upper left')
        
        ax5.set_title('Alpha Loss Value')
        ax5.plot(data['Gen'], data['Alpha Loss'], label='Alpha Loss', alpha=0.2, color='white')
        ax5.plot(data['Gen'], data['Running Alpha Loss'], label='Running Alpha Loss', color='red')
        ax5.legend(loc='upper left')
        
        ax6.set_title('Alpha Value')
        ax6.plot(data['Gen'], data['Alpha'], label='Alpha', color='red')
        ax6.legend(loc='upper left')
        
        fig.canvas.draw()

    ani = FuncAnimation(fig, animate, interval=1000)


    plt.tight_layout(pad=2)
    plt.show()