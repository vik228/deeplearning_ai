import matplotlib.pyplot as plt

def show_image(image,subplots = plt,**kwargs):
    subplots.imshow(image, **kwargs)

def show_image_in_grid(images, rows=2, cols=10, **kwargs):
    f, axarr = plt.subplots(3, 165)
    curr_row = 0
    index = 0
    for image in images:
        col = index % 3
        axarr[col,curr_row].imshow(image)
        if col == 2:
         # we have finished the current row, so increment row counter
         curr_row += 1
        index += 1


