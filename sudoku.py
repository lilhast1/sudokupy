#!/usr/bin/env python3
import cv2
import numpy as np 
import sys
from classify import classify
from keras.preprocessing.image import img_to_array
import math
from scipy import ndimage

BLACK = [0, 0, 0]



def correction(topology, digit, pvector):
	if topology == 0:
		return 0
	if topology == 1:
		arr = np.array([0, pvector[0, 1], pvector[0, 2], pvector[0, 3], 
						pvector[0, 4], pvector[0, 5], 0, pvector[0, 7],
						0, 0])
		return np.argmax(arr)
		[p2, 1, 2, 3, 4, 5, 5, 7, p2, p2][digit]
	if topology == 2:
		arr = np.array([0, 0, 0, 0, pvector[0, 4], 
						0, pvector[0, 6], 0, 0, pvector[0, 9]])
		return np.argmax(arr)
	if topology == 3:
		return 8
	return digit

def load_model(path):
	return cv2.ml.KNearest.load(path)

knn = load_model('ocr.knn')

def read(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (640, 480))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img, gray

def preprocess(gray):
	g = cv2.GaussianBlur(gray, (5, 5), 0)
	return cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

def find_square(thresh):
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	biggest = None
	max_area = 0
	for c in contours:
		area = cv2.contourArea(c)
		if area > thresh.size / 4:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			if area > max_area and len(approx) == 4:
				biggest = approx
				max_area = area
	return biggest, max_area

def rectify(h):
	h = h.reshape((4, 2))
	hnew = np.zeros((4, 2), dtype=np.float32)

	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]

	diff = np.diff(h, axis=1)
	hnew[1] = h[np.argmin(diff)]
	hnew[3] = h[np.argmax(diff)]

	return hnew

def warp(gray, approx):
	h = np.array([ [0,0], [449,0], [449,449], [0,449]], np.float32)
	r = cv2.getPerspectiveTransform(approx, h)
	return cv2.warpPerspective(gray, r, (450, 450))

def remove_grid(warpped):
	g = cv2.GaussianBlur(warpped, (5, 5), 0)
	bw = cv2.adaptiveThreshold(g, 255, 1, 1, 11, 2)
	horizontal = bw.copy()
	vertical = bw.copy()

	cols = horizontal.shape[1]
	horizontal_size = cols // 12

	horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
	horizontal = cv2.erode(horizontal, horizontal_structure)
	horizontal = cv2.dilate(horizontal, horizontal_structure)

	rows = vertical.shape[0]
	vertical_size = rows // 12

	vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
	vertical = cv2.erode(vertical, vertical_structure)
	vertical = cv2.dilate(vertical, vertical_structure)

	result = bw - vertical - horizontal

	kernel = np.ones((2,2), np.uint8)
	result = cv2.erode(result, kernel)
	result = cv2.dilate(result, kernel)

	return result


def center_digit(resized):
	big = cv2.copyMakeBorder(resized, 28, 28, 28, 28, cv2.BORDER_CONSTANT, value=BLACK)
	ret, binary_big = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	contours, hierarchy = cv2.findContours(binary_big, 1, 2)
	if len(contours) == 0:
		return False, None
	cnt = contours[0]
	cnt = max(contours, key=lambda c: cv2.contourArea(c))
	
	M = cv2.moments(cnt)
	# if len(contours) > 1 or M['m00'] < 1e-5:
	# 	print(f'wtf {len(contours)} {M["m00"]}')
	# 	return False, None
	cv2.imshow('bbig', binary_big)
	cx = int(M['m10'] / M['m00'])
	cy = int(M['m01'] / M['m00'])

	return True, big[(cy-10):(cy+10), (cx-10):(cx+10)], len(contours)

def get_shift(img):
	cy, cx = ndimage.measurements.center_of_mass(img)
	rows, cols = img.shape
	return np.round(cols / 2.0 - cx).astype(int), np.round(rows / 2.0 - cy).astype(int)

def shift(img, sx, sy):
	rows, cols = img.shape
	M = np.float32([[1,0,sx],[0,1,sy]])
	return cv2.warpAffine(img, M, (cols, rows))

def normalize_digit(img):
	thresh, gray = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	while gray.shape[0] > 1 and np.sum(gray[0]) == 0:
		gray = gray[1:]
	while gray.shape[1] > 1 and np.sum(gray[:,0]) == 0:
		gray = np.delete(gray, 0, 1)
	while gray.shape[0] > 1 and np.sum(gray[-1]) == 0:
		gray = gray[:-1]
	while gray.shape[1] > 1 and np.sum(gray[:,-1]) == 0:
		gray = np.delete(gray, -1, 1)

	rows, cols = gray.shape 
	if rows <= 1 or cols <= 1:
		return False, None, 0

	if rows > cols:
		factor = 20.0 / rows
		rows = 20	
		cols = int(round(cols * factor))
	else:
		factor = 20.0 / rows
		cols = 20	
		rows = int(round(rows * factor))
	gray = cv2.resize(gray, (cols, rows))

	colsPadding = (int(math.ceil((28-cols) / 2.0)), 
					int(math.floor((28-cols) / 2.0)))
	rowsPadding = (int(math.ceil((28-rows) / 2.0)), 
					int(math.floor((28-rows) / 2.0)))
	gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

	shiftx, shifty = get_shift(gray)

	ret, binary_big = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	contours, hierarchy = cv2.findContours(binary_big, 1, 2)

	return True, shift(gray, shiftx, shifty), len(contours)

def get_digits(img):
	digits = []
	height, width  = img.shape[0], img.shape[1]
	cell_w = width // 9
	cell_h = height // 9
	target_w = 28
	target_h = 28

	for i in range(9):
		row = []
		for j in range(9):
			cell = img[(cell_h * i): (cell_h * (i + 1)), (cell_w * j):(cell_w * (j + 1))]
			resized = cv2.resize(cell, (target_w, target_h))

			# cv2.imshow('resized', resized)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			found, digit_img, topology = normalize_digit(resized)
			if found == False:
				print(f'cell: {i*9 + j} | topology: {topology} is empty')
				row.append(0)
				continue

			# input_data = digit_img.reshape(-1, 400).astype(np.float32)
			# ret, result, neighbours, dist = knn.findNearest(input_data, k = 4)

			# print(f'cell {i*9 + j}: classified as: {int(ret)} with dist {dist} neighbours {neighbours} and is: {int(ret) == RESULT[i][j]} should be: {RESULT[i][j]}')

			# row.append(int(result[0, 0]))

			normalized = digit_img.astype('float32') / 255.0
			input_data = np.expand_dims(normalized, axis=0)
			digit, pvec = classify(input_data)
			digit = correction(topology, digit, pvec)
			row.append(digit)
			print(f'cell: {i*9 + j} | topology: {topology} digit: {row[-1]}')
			# if suma > target_h * target_w / 10:
			# 	# vec_f = cv2.normalize(vec, None, alpha=0, beta=1, 
			# 	# 					norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			# 	vec_f = vec / 255
			# 	tensor = np.expand_dims(vec_f, axis=0)
				
			# else:
			# 	row.append(0)
		digits.append(row)
	return digits

def postprocess(clear, warpped):
	masked = cv2.bitwise_and(warpped, clear)

	contrasted = cv2.equalizeHist(masked)
	
	log = contrasted.copy()
	cv2.intensity_transform.logTransform(contrasted, log)

	contours, hierarchy = cv2.findContours(log, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	mask = np.zeros_like(log)
	for c in contours:
		area = cv2.contourArea(c)
		if area > 20:
			cv2.fillPoly(mask,[c], 255)
	return cv2.bitwise_and(mask, log)

def findNextCell(grid):
    for x in range(0, 9):
        for y in range(0, 9):
            if grid[x][y] == 0:
                return x, y
    return -1, -1

def correct(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            secTopX, secTopY = 3 *(i//3), 3 *(j//3)
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == e:
                        return False
            return True
    return False

def solve(grid, i=0, j=0):
    i, j = findNextCell(grid)
    if i == -1:
        return True

    for e in range(1, 10):
        #Try different values in i, j location
        if correct(grid, i, j, e):
            grid[i][j] = e
            if solve(grid, i, j):
                return True
            grid[i][j] = 0

    return False

def printSudokuLine(arr):
	s = "|"
	for i in range(3):
		for j in range(3):
			s += str(arr[j + 3 * i]) + " "
		s += "|"
	print(s)

def printSudoku(grid):
	print("-" * 11 * 2)
	for i in range(9):
		printSudokuLine(grid[i])
		if i % 3 == 2:
			print("-" * 11 * 2)

def main():
	img, gray = read(sys.argv[1])

	cv2.imshow('original', img)

	thresh = preprocess(gray)

	approx, max_area = find_square(thresh)

	approx = rectify(approx)

	warpped = warp(gray, approx) 

	# remove all grid lines
	clear = remove_grid(warpped)

	puzzle = postprocess(clear, warpped)

	puzzle = cv2.GaussianBlur(puzzle, (5, 5), 0)

	grid = get_digits(puzzle)

	solve(grid)
	printSudoku(grid)


	cv2.imshow('theimage', puzzle)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__=='__main__':
	main()