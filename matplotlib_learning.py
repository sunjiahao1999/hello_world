import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(-1,1,50)
# y1 = x ** 2 - 1
# y2 = 2 * x + 1
#
# plt.figure(num=2, figsize=(4, 4))
# l1, = plt.plot(x, y1, label='up')          #注意l1要加逗号
# l2, = plt.plot(x, y2, color='orange', linewidth=1.0, linestyle='--')
# plt.legend(handles=[l2], labels=['bbb'], loc='best')
#
# plt.xlim((-5, 5))
# plt.xlabel('x')
# new_ticks = np.linspace(-5, 5, 11)
# plt.xticks(new_ticks)
#
# plt.ylim((-5, 5))
# plt.ylabel('y')
# plt.yticks([-5, -2.5, 0, 2.5],
#            [r'$n$', r'$m$', r'$s$', r'$l\ \alpha$'])
#
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data', -1))
# ax.spines['left'].set_position(('data', 0))
#
# x0 = 1
# y0 = 2 * x0 + 1
# plt.scatter(x0, y0, s=50, color='b')
# plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)
#
# plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30), textcoords='offset points'
#              , fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
# plt.text(-3.7, 3, r'$\sigma_i\ \alpha_t$', fontdict={'size': 16, 'color': 'r'})
#
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(12)
#     label.set_bbox({'facecolor':'yellow','edgecolor':'None','alpha':0.3})
#
# plt.show()

#####################
# 散点图
# n = 1024
# x = np.random.normal(0, 1, n)
# y = np.random.normal(0, 1, n)
# t = np.arctan2(y, x)
# plt.scatter(x, y, s=75, c=t, alpha=0.5)
#
# plt.xlim((-2, 2))
# plt.ylim((-2, 2))
# plt.xticks(())
# plt.yticks(())
# plt.show()

###################
# 柱状图
# n = 12
# X = np.arange(n)
# Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
# Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
#
# plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
# plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
#
# for x, y in zip(X, Y1):
#     plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
# for x, y in zip(X, Y2):
#     plt.text(x, -y-0.05, '%.2f' % -y, ha='center', va='top')
#
# plt.xlim(-.5, n)
# plt.xticks(())
# plt.ylim(-1.25, 1.25)
# plt.yticks(())
# plt.show()

####################
# 等高线图
# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
# X, Y = np.meshgrid(x, y)
#
#
# def f(x, y):
#     return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
#
#
# plt.figure(figsize=(5, 4))
# plt.contourf(X, Y, f(X, Y), 8, alpha=0.7, cmap='hot')
# C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidths=0.5)
# plt.clabel(C, inline=True, fontsize=7)
# plt.xticks(())
# plt.yticks(())
# plt.show()
#################################
# 图片
# a = np.random.rand(3, 3)
# plt.imshow(a, interpolation='nearest', cmap='bone', origin='upper')
# plt.colorbar(shrink=0.9)
#
# plt.xticks()
# plt.yticks()
# plt.show()

##############################
# 三维图
#
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(5, 4))
ax = Axes3D(fig)
x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.contour(X, Y, Z, zdir='z', offset=-2, cmap='rainbow')
ax.set_zlim(-2, 2)
plt.show()
##############################
# SUBPLOT
# plt.subplot(2, 1, 1)
# plt.plot([0, 1], [1, 1])
# plt.subplot(223)
# plt.plot([2, 3], [2, 2])
# plt.subplot(224)
# plt.plot([0, 0], [2, 3])
# plt.show()
#################################
# SUBPLOT2GRID
# plt.figure()
# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
# ax1.plot([1, 2], [1, 2])
# ax2 = plt.subplot2grid((3, 3), (1, 0), rowspan=2)
# ax2.plot([1, 2], [1, 2])
# ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=2)
# ax3.plot([1, 2], [1, 2])
# ax4 = plt.subplot2grid((3, 3), (2, 1))
# ax4.plot([1, 2], [1, 2])
# ax5 = plt.subplot2grid((3, 3), (2, 2))
# ax5.plot([1, 2], [1, 2])
# plt.show()
####################################
# 图中图
#
# fig = plt.figure()
# x = [x for x in range(1, 8)]
# y = [1, 3, 4, 2, 5, 8, 6]
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 = fig.add_axes([left, bottom, width, height])
# ax1.plot(x, y, 'r')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('title')
# ax2 = fig.add_axes([0.2, 0.6, 0.2, 0.2])
# ax2.plot(y, x, 'g')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_title('title')
# ax3 = fig.add_axes([0.65, 0.2, 0.2, 0.2])
# ax3.plot(y[::-1], x, 'b')
# ax3.set_xlabel('x')
# ax3.set_ylabel('y')
# ax3.set_title('title')
# plt.show()
####################
# 次坐标
# x = np.arange(0, 10, 0.1)
# y1 = 0.05 * x ** 2
# y2 = -y1
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(x, y1)
# ax2.plot(x, y2, 'orange')
# plt.show()
############################
# 动画
# from matplotlib import animation
#
# fig, ax = plt.subplots()
# x = np.arange(0, 2 * np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))
#
#
# def animate(i):  # 每过一帧调用一次该函数，且i自增1，增长到frames为止
#     line.set_ydata(np.sin(x + i / 100.0))
#     return line
#
#
# def init():
#     line.set_ydata(np.sin(x))
#     return line
#
#
# ani = animation.FuncAnimation(fig=fig, func=animate, frames=1000, init_func=init,
#                               interval=1, blit=False)
# plt.show()
#####################################
# gold, chihh = 400, 400
# gold_height = 40 + 10 * np.random.randn(gold)
# chihh_height = 25 + 6 * np.random.randn(chihh)
#
# plt.hist([gold_height, chihh_height],stacked=True, color=['r', 'b'])
# plt.show()
