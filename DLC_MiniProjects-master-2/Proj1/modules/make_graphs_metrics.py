#!/usr/bin/env python
# coding: utf-8

# In[178]:


import matplotlib.pyplot as plt
import matplotlib
import itertools
import numpy as np
import matplotlib.lines as mlines


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
 

figure = plt.figure(figsize=(3,3))

train_time_sum = [10.82,5.04,20.15,3.76,15.14,5.01,11.71,16.46,12.42,5.50,16.35,13.56,9.08,5.90,49.56,39.27,46.56,41.56,45.36,36.08,46.40,33.86]
mean_accu = [97.33,96.45,84.24,85.01,96.35,95.29,80.95,77.61,96.88,96.27,96.88,96.27,95.34,95.73,95.17,84.17,94.14,83.66,96.10,87.47,96.51,85.39]

markers = ['*', '^','o']
colors = ['r','g','b']

I1 = plt.scatter(train_time_sum[0],mean_accu[0],color=colors[1],marker=markers[1],s=150, label='Individual with batch norm')
I2 = plt.scatter(train_time_sum[1],mean_accu[1],color=colors[1],marker=markers[1],s=150, label='Individual without batch norm') 
ST1 = plt.scatter(train_time_sum[2],mean_accu[2],color=colors[1],marker=markers[1],s=150, label='Siamese (target) with batch norm')
ST2 = plt.scatter(train_time_sum[3],mean_accu[3],color=colors[1],marker=markers[1],s=150, label='Siamese (target) with batch norm')
SN1 = plt.scatter(train_time_sum[4],mean_accu[4],color=colors[1],marker=markers[0],s=150, label='Siamese no sharing with batch norm') 
SN2 = plt.scatter(train_time_sum[5],mean_accu[5],color=colors[1],marker=markers[0],s=150, label='Siamese no sharing without batch norm') 
SP1 = plt.scatter(train_time_sum[6],mean_accu[6],color=colors[0],marker=markers[0],s=150, label='SimpleNet with batch norm') 
SP2 = plt.scatter(train_time_sum[7],mean_accu[7],color=colors[0],marker=markers[0],s=150, label='SimpleNet without batch norm') 
SC1 = plt.scatter(train_time_sum[8],mean_accu[8],color=colors[1],marker=markers[1],s=150, label='Siamese (class) with batch norm') 
SC2 = plt.scatter(train_time_sum[9],mean_accu[9],color=colors[1],marker=markers[1],s=150, label='Siamese (class) without batch norm') 
SA1 = plt.scatter(train_time_sum[10],mean_accu[10],color=colors[1],marker=markers[2],s=150, label='Siamese (auxloss) with batch norm') 
SA2 = plt.scatter(train_time_sum[11],mean_accu[11],color=colors[1],marker=markers[2],s=150, label='Siamese (auxloss) without batch norm') 
SC1 = plt.scatter(train_time_sum[12],mean_accu[12],color=colors[1],marker=markers[2],s=150, label='Siamese (contrastive loss) with batch norm') 
SC2 = plt.scatter(train_time_sum[13],mean_accu[13],color=colors[1],marker=markers[2],s=150, label='Siamese (contrastive loss) without batch norm')
RN1 = plt.scatter(train_time_sum[14],mean_accu[14],color=colors[2],marker=markers[0],s=150, label='ResNet (no sharing) with batch norm')
RN2 = plt.scatter(train_time_sum[15],mean_accu[15],color=colors[2],marker=markers[0],s=150, label='ResNet (no sharing) without batch norm')
RN3 = plt.scatter(train_time_sum[16],mean_accu[16],color=colors[2],marker=markers[0],s=150, label='ResNet (no sharing) with batch norm with dropout')
RN4 = plt.scatter(train_time_sum[17],mean_accu[17],color=colors[2],marker=markers[0],s=150, label='ResNet (no sharing) without batch norm with dropout')
RS1 = plt.scatter(train_time_sum[18],mean_accu[18],color=colors[2],marker=markers[1],s=150, label='ResNet (sharing) with batch norm')
RS2 = plt.scatter(train_time_sum[19],mean_accu[19],color=colors[2],marker=markers[1],s=150, label='ResNet (sharing) without batch norm')
RS3 = plt.scatter(train_time_sum[20],mean_accu[20],color=colors[2],marker=markers[1],s=150, label='ResNet (sharing) with batch norm with dropout')
RS4 = plt.scatter(train_time_sum[21],mean_accu[21],color=colors[2],marker=markers[1],s=150, label='ResNet (sharing) without batch norm with dropout')



lab1 = ['No weight sharing','Weight sharing','Weight sharing + auxillary losses']
lab2 = ['Simple CNN', 'Siamese CNN', 'ResNet']

star = mlines.Line2D([], [], marker='*', linestyle='None',
                          markersize=10)
triangle = mlines.Line2D([], [], marker='^', linestyle='None',
                          markersize=10)
circle = mlines.Line2D([], [],  marker='o', linestyle='None',
                          markersize=10)
square1 = mlines.Line2D([], [],  marker='s', color='r', linestyle='None',
                          markersize=10)
square2 = mlines.Line2D([], [],  marker='s', color='g', linestyle='None',
                          markersize=10)
square3 = mlines.Line2D([], [],  marker='s', color='b', linestyle='None',
                          markersize=10)

legend1 = plt.legend((star,triangle,circle),lab1,markerscale=0.4, scatterpoints=1, fontsize=7,loc='center left', bbox_to_anchor=(1,0.6),fancybox=True)
legend2 = plt.legend((square1,square2,square3),lab2,title='Cluster families',markerscale=0.4, scatterpoints=1, fontsize=7,loc='center left', bbox_to_anchor=(1,0.3),fancybox=True)
legend1.set_title('Techniques',prop={'size':9})
legend2.set_title('Cluster families',prop={'size':9})

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
plt.xlabel('Mean training time [s]') 
plt.ylabel('Mean testing accuracy [%]')
plt.grid(True)
plt.show()

plt.savefig('accuracies.pgf',bbox_extra_artists=(legend1,legend2), bbox_inches='tight')


# In[179]:


figure = plt.figure(figsize=(3,3))

train_time_sum = [10.82,5.04,20.15,3.76,15.14,5.01,11.71,16.46,12.42,5.50,16.35,13.56,9.08,5.90,49.56,39.27,46.56,41.56,45.36,36.08,46.40,33.86]
params = [62514,61706,62386,61938,124308,1234412,298082,297378,62154,61706,360236,359084,62514,61706,47444,46804,47444,46804,23722,23402,23722,23402]

markers = ['*', '^','o']
colors = ['r','g','b']

I1 = plt.scatter(train_time_sum[0],params[0],color=colors[1],marker=markers[1],s=150, label='Individual with batch norm')
I2 = plt.scatter(train_time_sum[1],params[1],color=colors[1],marker=markers[1],s=150, label='Individual without batch norm') 
ST1 = plt.scatter(train_time_sum[2],params[2],color=colors[1],marker=markers[1],s=150, label='Siamese (target) with batch norm')
ST2 = plt.scatter(train_time_sum[3],params[3],color=colors[1],marker=markers[1],s=150, label='Siamese (target) with batch norm')
SN1 = plt.scatter(train_time_sum[4],params[4],color=colors[1],marker=markers[0],s=150, label='Siamese no sharing with batch norm') 
SN2 = plt.scatter(train_time_sum[5],params[5],color=colors[1],marker=markers[0],s=150, label='Siamese no sharing without batch norm') 
SP1 = plt.scatter(train_time_sum[6],params[6],color=colors[0],marker=markers[0],s=150, label='SimpleNet with batch norm') 
SP2 = plt.scatter(train_time_sum[7],params[7],color=colors[0],marker=markers[0],s=150, label='SimpleNet without batch norm') 
SC1 = plt.scatter(train_time_sum[8],params[8],color=colors[1],marker=markers[1],s=150, label='Siamese (class) with batch norm') 
SC2 = plt.scatter(train_time_sum[9],params[9],color=colors[1],marker=markers[1],s=150, label='Siamese (class) without batch norm') 
SA1 = plt.scatter(train_time_sum[10],params[10],color=colors[1],marker=markers[2],s=150, label='Siamese (auxloss) with batch norm') 
SA2 = plt.scatter(train_time_sum[11],params[11],color=colors[1],marker=markers[2],s=150, label='Siamese (auxloss) without batch norm') 
SCO1 = plt.scatter(train_time_sum[12],params[12],color=colors[1],marker=markers[2],s=150, label='Siamese (contrastive loss) with batch norm') 
SCO2 = plt.scatter(train_time_sum[13],params[13],color=colors[1],marker=markers[2],s=150, label='Siamese (contrastive loss) without batch norm')
RN1 = plt.scatter(train_time_sum[14],params[14],color=colors[2],marker=markers[0],s=150, label='ResNet (no sharing) with batch norm')
RN2 = plt.scatter(train_time_sum[15],params[15],color=colors[2],marker=markers[0],s=150, label='ResNet (no sharing) without batch norm')
RN3 = plt.scatter(train_time_sum[16],params[16],color=colors[2],marker=markers[0],s=150, label='ResNet (no sharing) with batch norm with dropout')
RN4 = plt.scatter(train_time_sum[17],params[17],color=colors[2],marker=markers[0],s=150, label='ResNet (no sharing) without batch norm with dropout')
RS1 = plt.scatter(train_time_sum[18],params[18],color=colors[2],marker=markers[1],s=150, label='ResNet (sharing) with batch norm')
RS2 = plt.scatter(train_time_sum[19],params[19],color=colors[2],marker=markers[1],s=150, label='ResNet (sharing) without batch norm')
RS3 = plt.scatter(train_time_sum[20],params[20],color=colors[2],marker=markers[1],s=150, label='ResNet (sharing) with batch norm with dropout')
RS4 = plt.scatter(train_time_sum[21],params[21],color=colors[2],marker=markers[1],s=150, label='ResNet (sharing) without batch norm with dropout')


lab1 = ['No weight sharing','Weight sharing','Weight sharing + auxillary losses']
lab2 = ['Simple CNN', 'Siamese CNN', 'ResNet']

star = mlines.Line2D([], [], marker='*', linestyle='None',
                          markersize=10)
triangle = mlines.Line2D([], [], marker='^', linestyle='None',
                          markersize=10)
circle = mlines.Line2D([], [],  marker='o', linestyle='None',
                          markersize=10)
square1 = mlines.Line2D([], [],  marker='s', color='r', linestyle='None',
                          markersize=10)
square2 = mlines.Line2D([], [],  marker='s', color='g', linestyle='None',
                          markersize=10)
square3 = mlines.Line2D([], [],  marker='s', color='b', linestyle='None',
                          markersize=10)

legend1 = plt.legend((star,triangle,circle),lab1,markerscale=0.4, scatterpoints=1, fontsize=7,loc='center left', bbox_to_anchor=(1,0.6),fancybox=True)
legend2 = plt.legend((square1,square2,square3),lab2,title='Cluster families',markerscale=0.4, scatterpoints=1, fontsize=7,loc='center left', bbox_to_anchor=(1,0.3),fancybox=True)
legend1.set_title('Techniques',prop={'size':9})
legend2.set_title('Cluster families',prop={'size':9})

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
plt.xlabel('Mean training time [s]') 
plt.ylabel('Number of parameters')
plt.grid(True)
plt.show()


plt.savefig('params.pgf',bbox_extra_artists=(legend1,legend2), bbox_inches='tight')


# In[ ]:




