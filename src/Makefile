TARGETS := test_nb.ipynb

$(TARGETS): %.ipynb : %.py
	python3 py2ipynb.py $< $@

clean:
	rm -f $(TARGETS)

.PHONY: clean
