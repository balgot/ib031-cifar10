TARGETS := test_nb.ipynb first_presentation_nb.ipynb CIFAR10_final.ipynb

all: $(TARGETS)

$(TARGETS): %.ipynb : %.py
	python3 py2ipynb.py $< $@

clean:
	rm -f $(TARGETS)

.PHONY: clean
