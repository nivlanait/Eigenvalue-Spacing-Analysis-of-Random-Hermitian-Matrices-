# ===== CONFIG =====
PYTHON = python3
SCRIPT = main.py
INPUT = hermitian_matrix_200.npy

# ===== DEFAULT =====
run:
	$(PYTHON) $(SCRIPT)

# ===== RANDOM MATRIX EXPERIMENT =====
random:
	$(PYTHON) $(SCRIPT) --n 150 --trials 60

# ===== INPUT MATRIX =====
input:
	$(PYTHON) $(SCRIPT) --input $(INPUT)

# ===== INPUT (NO PLOT) =====
input-noplot:
	$(PYTHON) $(SCRIPT) --input $(INPUT) --no-plot

# ===== CUSTOM PARAMS =====
custom:
	$(PYTHON) $(SCRIPT) --n 200 --trials 100 --bins 80

# ===== INSTALL DEPENDENCIES =====
install:
	pip install numpy matplotlib

# ===== CLEAN =====
clean:
	rm -rf __pycache__ *.pyc