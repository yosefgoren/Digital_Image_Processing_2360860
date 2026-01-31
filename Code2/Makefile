RESULTS_IDX?=6
STUDENT_1_ID=214146896
STUDENT_2_ID=211515606
RESULTS_DIR=unet_sidd/results/$(RESULTS_IDX)

Q2_PDF_PARTS=q2_intro.pdf $(RESULTS_DIR)/report.pdf $(RESULTS_DIR)/test_results.pdf
FULL_SOL_PDF_PARTS=sol_intro.pdf q1_sol.pdf $(Q2_PDF_PARTS) StyleTransfer.pdf 

TARGET_ZIP=code2_$(STUDENT_1_ID)_$(STUDENT_2_ID).zip
GENERATED_FILES=sol_intro.pdf q2_intro.pdf StyleTransfer.pdf $(TARGET_ZIP)

ATTACHED_SOURCE_FILES=./image_utils.py \
	./q1_sol.py \
	./style_transfer.py \
	./unet_sidd/datasets/sidd.py \
	./unet_sidd/generate_report.py \
	./unet_sidd/metrics.py \
	./unet_sidd/models/unet.py \
	./unet_sidd/test.py \
	./unet_sidd/train.py \
	./unet_sidd/utils.py

ATTACHED_AUX_FILES=./Makefile \
	$(RESULTS_DIR)/unet_sidd.pth \
	$(RESULTS_DIR)/metrics.json \
	StyleTransfer.ipynb

all: $(TARGET_ZIP)

StyleTransfer.pdf: StyleTransfer.ipynb
	jupyter nbconvert --to pdf $<

%.pdf: %.md
	pandoc $< -o $@ -V pagestyle=empty

full_sol.pdf: $(FULL_SOL_PDF_PARTS)
	pdfunite $^ $@

$(TARGET_ZIP): full_sol.pdf $(ATTACHED_SOURCE_FILES) $(ATTACHED_AUX_FILES)
	zip -r $@ $^

clean:
	rm -f $(GENERATED_FILES)