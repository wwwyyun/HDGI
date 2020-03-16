import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()


    def forward(self, seq, msk):
        data = torch.squeeze(seq)
        np_data = data.cpu().detach().numpy()
        k = 7
        cl, c = self.k_means(np_data, k)
        #c = torch.from_numpy(c).float().to('cuda:0')
        #centroids = c.unsqueeze(0)
        if msk is None:
            return torch.from_numpy(c), torch.from_numpy(cl)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(centroids * msk, 1) / torch.sum(msk)


    def k_means(self, data, k):

        # feature normalization (feature scaling)
        X_scaler = StandardScaler()
        x = X_scaler.fit_transform(data)

        # PCA
        pca = PCA(n_components=32)  #降到32维
        pca.fit(x)
        newX = pca.transform(x)
        # print(pca.explained_variance_ratio_)
        # print(pca.explained_variance_)
        print(newX)
        print(newX.shape)


        estimator = KMeans(n_clusters=k)#构造聚类器
        #print(data)
        estimator.fit(newX)#聚类
        label_pred = estimator.labels_ #获取聚类标签
        #print(data.shape[1])
        '''
        centroids = np.zeros((k, data.shape[1]))
        #print(centroids[:2, :3])
        #print(np.mean(np.array(np.array([data[j] for j in range(10) if label_pred[j]==0]))))
        
        for i in range(k):
            temp = []
            for j in range(data.shape[1]):
                if label_pred[j] == i:
                    temp.append(data[j])

            #a = np.array(temp)
            #print(a.shape)
            #print(np.mean(temp, axis=0))
            if temp == []:
                centroids[i][:] = np.zeros(data.shape[1])
            else:
                centroids[i][:] = np.mean(temp, axis=0)
           '''
        centroids = estimator.cluster_centers_ #获取聚类中心
        #inertia = estimator.inertia_ # 获取聚类准则的总和

        c = pca.inverse_transform(centroids) #变原来的维度
        print(c.shape)
        return label_pred, c
