import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt

"""
    Harflerin ve 4 bit hatası yapılmış versiyonlarının oluşturulması
    **Verimizdeki 0 değerlerini 0.1; 1 değerlerini ise 0.9'a alarak düzenledik.
    Bu sayede 0.1 değerleri ağırlıklarla çarpıldığı zaman daha anlamlı sonuçlar bulunacak.
    t = np.where(t == 0, 0.1, 0.9)
 """


def pattern_designer():
    t = np.ones((5, 10))
    t[:, 5] = 0
    t[0, :] = 0
    t = np.where(t == 0, 0.1, 0.9)
    t_degisik = np.copy(t)
    t_degisik[0, 0], t_degisik[1, 0], t_degisik[3,
                                                4], t_degisik[3, 5] = 0.9, 0.1, 0.1, 0.9
    l = np.ones((5, 10))
    l[:, 0] = 0
    l[-1, :] = 0
    l = np.where(l == 0, 0.1, 0.9)
    l_degisik = np.copy(l)
    l_degisik[4, 4], l_degisik[3, 4], l_degisik[4,
                                                9], l_degisik[3, 9] = 0.9, 0.1, 0.9, 0.1
    h = np.ones((5, 10))
    h[:, 2] = 0
    h[:, 6] = 0
    h[2, :] = 0
    h = np.where(h == 0, 0.1, 0.9)
    h_degisik = np.copy(h)
    h_degisik[0, 8], h_degisik[2, 6], h_degisik[0,
                                                0], h_degisik[2, 2] = 0.1, 0.9, 0.1, 0.9
    a = np.ones((5, 10))
    a[:, 9] = 0
    a[0, :] = 0
    a = np.where(a == 0, 0.1, 0.9)
    a_degisik = np.copy(a)
    a_degisik[0, 2], a_degisik[1, 2], a_degisik[3,
                                                8], a_degisik[3, 9] = 0.9, 0.1, 0.1, 0.9
    return t, h, a, l, t_degisik, h_degisik, a_degisik, l_degisik

# harfe gürültü ekleme


def gray_noise(t, h, a, l):
    t1 = t + np.random.normal(0, .05, t.shape)
    h1 = h + np.random.normal(0, .05, h.shape)
    a1 = a + np.random.normal(0, .05, a.shape)
    l1 = l + np.random.normal(0, .05, l.shape)
    return t1, h1, a1, l1

# Veri kümelerini eğitime sokmaan önce vektör haline getirme


def data_vectorizer(data_to_vector):
    data_to_vector = [a.reshape((50, 1)) for a in data_to_vector]
    return data_to_vector


# veri oluşturan ve gürültü ekleyen fonksiyonlar çağrıldı.
t, h, a, l, t_degisik, h_degisik, a_degisik, l_degisik = pattern_designer()
t_noisy, h_noisy, a_noisy, l_noisy = gray_noise(t, h, a, l)

# her bir patter birbirine dik ve eşit uzaklıkta vektörler belirlendi
e1 = [1, 0, 0, 0]
e2 = [0, 1, 0, 0]
e3 = [0, 0, 1, 0]
e4 = [0, 0, 0, 1]

x_input = [t, h, a, l, t_degisik, h_degisik, a_degisik,
           l_degisik, t_noisy, h_noisy, a_noisy, l_noisy]
""" for i in x_input:
    plt.imshow(i, cmap='gray')
    plt.show() """

# eğitim kümesindeki veriler sınıflandırıldı ve vektörleştirildi
y_desired = [e1, e2, e3, e4, e1, e2, e3, e4, e1, e2, e3, e4]
x_input = data_vectorizer(x_input)


# test kümesi oluşturuldu ve vektörleştirildi
t_noisy_test, h_noisy_test, a_noisy_test, l_noisy_test = gray_noise(t, h, a, l)
x_test = data_vectorizer(
    [t, h, a, l, t_noisy_test, h_noisy_test, a_noisy_test, l_noisy_test])
y_desired_test = [e1, e2, e3, e4, e1, e2, e3, e4]

for i in range(len(x_input)):
    print(x_input[i])
    print(x_input[i].shape)
# Veriler tekrar kullanılmak için kaydedildi
np.save("x_egitim", x_input)
np.save("yd_egitim", y_desired)
np.save("x_test", x_test)
np.save("yd_test", y_desired_test)

""" for i in x_input:
    plt.imshow(i, cmap='gray')
    plt.show()
 """
