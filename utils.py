import numpy as np
from sklearn import linear_model

def get_clrchkr(img, mask):
    return (img[int(mask[0, 1]):int(mask[0,1] + mask[0,3]), 
                      int(mask[0,0]):int(mask[0,0] + mask[0,2])]).copy()

def get_patch_color(clrchkr, patch_polygon_x, patch_polygon_y):
    x1 = np.amax(patch_polygon_x[0:2])
    x2 = np.amin(patch_polygon_x[2:4])
    y2 = np.amin(patch_polygon_y[1:3])
    y1 = np.max([patch_polygon_y[0], patch_polygon_y[3]])
    patch = clrchkr[int(y1):int(y2), int(x1):int(x2)]
    return np.array([(patch[:,:,0]).mean(), (patch[:,:,1]).mean(), (patch[:,:,2]).mean()])

def wb(img, mask):
    img = img.copy()
    print("WB:", get_patch_color(img, mask[0,0] + mask[1], mask[0,1] + mask[2]))
    img /= get_patch_color(img, mask[0,0] + mask[1], mask[0,1] + mask[2])
    return img

def get_pallete(clrchkr, mask):
    palette = np.zeros((24, 24, 3))
    for i in range(palette.shape[0]):
        palette[(i // 6 * 6):((i // 6 + 1) * 6), (i % 6 * 4):((i % 6 +1)*4)] = get_patch_color(clrchkr, mask[2 * i + 1], mask[2 * i + 2])
    return palette

def get_colors_from_patches(clrchkr, mask):
    colors = []
    for i in range((mask.shape[0] - 1) // 2):
        colors.append(get_patch_color(clrchkr, mask[i*2+1], mask[i*2+2]))
    return np.array(colors)

#A*src=res
def calcA(src, res):
    c = np.concatenate(res)
    z = np.zeros((3))
    C_patches = []
    for patch in src:
        C_patches.append(np.array( [np.concatenate((patch,z,z)), np.concatenate((z,patch,z)), np.concatenate((z,z,patch))] ))
    C = np.concatenate(C_patches)

    a = np.dot(np.dot(np.linalg.inv(np.dot(C.T, C)), C.T), c)
    
    A = np.array([a[0:3], a[3:6], a[6:9]])
    return A   

def fitA(X, y):
    y_r = y[:, 0]
    y_g = y[:, 1]
    y_b = y[:, 2]
    r_clsf = linear_model.LinearRegression()
    g_clsf = linear_model.LinearRegression()
    b_clsf = linear_model.LinearRegression()
    r_clsf.fit(X, y_r)
    g_clsf.fit(X, y_g)
    b_clsf.fit(X, y_b)
    A = np.zeros((3, 3))
    A[0] = r_clsf.coef_
    A[1] = g_clsf.coef_
    A[2] = b_clsf.coef_
    return A

def transform(A, img):
    img_t = img.copy()
    h = img_t.shape[0]
    l = img_t.shape[1]
    img_t = np.reshape((np.dot(A, (np.reshape(img_t, (h * l, 3))).T)).T, (h, l, 3))
    return img_t

def angdiff(img1, img2):
    diff = img1[:,:,0] * img2[:,:,0] + img1[:,:,1] * img2[:,:,1] + img1[:,:,2] * img2[:,:,2]
    diff =diff / np.sqrt(img1[:,:,0] ** 2 + img1[:,:,1] ** 2 + img1[:,:,2] ** 2 + 0.000001)
    diff = diff / np.sqrt(img2[:,:,0] ** 2 + img2[:,:,1] ** 2 + img2[:,:,2] ** 2 + 0.000001)
    diff = 180 * np.arccos(np.round(diff, 6)) / np.pi
    diffpic = np.zeros(img1.shape)
    diffpic[:,:,0] = diff
    diffpic[:,:,1] = diff
    diffpic[:,:,2] = diff
    return diff

if __name__ == "__main__":
    pass
