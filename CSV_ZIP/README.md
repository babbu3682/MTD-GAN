# CSV files for evaluating statistical significance with p-values using a paired t-test.

<p align="center"><img width="60%" src="figures/Brain_Total.png" /></p>

**Fig. A.** Comparison with previous methods on the brain test set for statistical significance evaluation, with p-values evaluated between MTD-GAN and other methods using a paired t-test. MTD-GAN demonstrated significantly better performance (P < 0.0001) compared to all previous works in all metrics, except for CTformer in terms of PSNR (p-value: 0.0255).


###

###

###


<p align="center"><img width="60%" src="figures/Abdomen_Total.png" /></p>

**Fig. B.** Comparison with previous methods on the abdomen test set for statistical significance evaluation, with p-values evaluated between MTD-GAN and other methods using a paired t-test. MTD-GAN demonstrated significantly better performance (P < 0.0001) compared to all previous works in all metrics.

###

As specified in "Section IV, B. Implementation Details" in our manuscirpt, the statistical significance of all experiments was assessed using the MedCalc software (Version 22.023). Paired t-tests were conducted for each metric, with the exception of the FID. For transparency, we are providing real-time captures of the MedCalc statistical analyses (refer to Fig. A and B). As evident in Fig. A and B, MTD-GAN demonstrates statistically significant performance differences compared to the previous works on all metrics, with p-values lower than 0.0001, except with CTformer in terms of PSNR (p-value: 0.0255). Additionally, we plan to publish CSV files containing scores for each metric across all test samples on GitHub [🖇️Link](https://github.com/babbu3682/MTD-GAN/tree/main/CSV_ZIP)
