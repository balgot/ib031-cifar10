CURRENT_DIR := $(shell pwd)
SRC_DIR := $(CURRENT_DIR)/src
TARGETS := test_nb.ipynb


$(TARGETS): %.ipynb : $(SRC_DIR)/%.py
	python3 $(SRC_DIR)/py2ipynb.py $< $(CURRENT_DIR)/$@

clean:
	rm -f *.ipynb

.PHONY: clean
