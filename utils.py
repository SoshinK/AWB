import numpy as np

def get_clrchkr(img, roi):
    return (img[int(roi[1]):int(roi[1] + roi[3]), 
                      int(roi[0]):int(roi[0] + roi[2])]).copy()

def get_patch_color(clrchkr, patch_polygon_x, patch_polygon_y):
    x1 = np.amax(patch_polygon_x[0:2])
    x2 = np.amin(patch_polygon_x[2:4])
    y2 = np.amin(patch_polygon_y[1:3])
    y1 = np.max([patch_polygon_y[0], patch_polygon_y[3]])
    patch = clrchkr[int(y1):int(y2), int(x1):int(x2)]
    return np.array([(patch[:,:,0]).mean(), (patch[:,:,1]).mean(), (patch[:,:,2]).mean()])

def wb(img, mask):
    img /= get_patch_color(img, mask[0,0] + mask[1], mask[0,1] + mask[2])
    return img / np.amax(img)

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

def transform(A, img):
    img_t = img.copy()
    h = img_t.shape[0]
    l = img_t.shape[1]
    img_t = np.reshape((np.dot(A, (np.reshape(img_t, (h * l, 3))).T)).T, (h, l, 3))
    return img_t


def angdiff(img1, img2):
    diff = np.zeros(img1.shape)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            diff[i, j, :] = np.arccos(round(np.dot(img1[i, j], img2[i, j]) / \
                np.linalg.norm(img1[i, j]) / np.linalg.norm(img2[i, j]), 9))
    return diff
