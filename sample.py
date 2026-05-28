import rvmn
import cv2

image = cv2.imread("C:/Users/siddh/Desktop/rvmn/1.png")
gray = rvmn.to_grayscale(image)

result = rvmn.apply_rvmn(gray, lam=0.1, rho=0.9, theta=0.3)
sobel = rvmn.apply_sobel(gray)
prewitt = rvmn.apply_prewitt(gray)

metrics = rvmn.compute_metrics(gray, result)
print(metrics.psnr, metrics.rmse)

search = rvmn.search_best_params(gray)

print(search.params)
print(search.metrics.psnr)
cv2.imwrite("rvmn_output.png", search.image)

comparison = rvmn.compare_edges(gray)

print(comparison.rvmn.params)
print(comparison.sobel_metrics.psnr)
print(comparison.prewitt_metrics.psnr)

