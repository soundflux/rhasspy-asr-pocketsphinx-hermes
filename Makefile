SHELL := bash
PYTHON_NAME = rhasspyasr_pocketsphinx_hermes
PACKAGE_NAME = rhasspy-asr-pocketsphinx-hermes
SOURCE = $(PYTHON_NAME)
PYTHON_FILES = $(SOURCE)/*.py *.py
SHELL_FILES = bin/$(PACKAGE_NAME) debian/bin/* scripts/*.sh *.sh
DOWNLOAD_DIR = download

PIP_INSTALL ?= install

.PHONY: reformat check venv dist sdist pyinstaller debian docker deploy

version := $(shell cat VERSION)
architecture := $(shell bash architecture.sh)

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

venv: downloads
	scripts/create-venv.sh

dist: sdist debian

test:
	echo "Skipping tests for now"

test-wavs:
	scripts/test_wavs.sh

sdist:
	python3 setup.py sdist

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker: pyinstaller
	docker build . -t "rhasspy/$(PACKAGE_NAME):$(version)" -t "rhasspy/$(PACKAGE_NAME):latest"

deploy:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin
	docker push "rhasspy/$(PACKAGE_NAME):$(version)"

# -----------------------------------------------------------------------------
# Debian
# -----------------------------------------------------------------------------

pyinstaller:
	scripts/build-pyinstaller.sh "${architecture}" "${version}"

debian:
	scripts/build-debian.sh "${architecture}" "${version}"

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

downloads: rhasspy-libs $(DOWNLOAD_DIR)/pocketsphinx-python.tar.gz

# Rhasspy development dependencies
rhasspy-libs: $(DOWNLOAD_DIR)/rhasspy-asr-pocketsphinx-0.1.4.tar.gz $(DOWNLOAD_DIR)/rhasspy-silence-0.1.2.tar.gz $(DOWNLOAD_DIR)/rhasspy-hermes-0.1.6.tar.gz

$(DOWNLOAD_DIR)/rhasspy-asr-pocketsphinx-0.1.4.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-asr-pocketsphinx/archive/master.tar.gz"

$(DOWNLOAD_DIR)/rhasspy-silence-0.1.2.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-silence/archive/master.tar.gz"

$(DOWNLOAD_DIR)/rhasspy-hermes-0.1.6.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-hermes/archive/master.tar.gz"

# Download Python Pocketsphinx library with no dependency on PulseAudio.
$(DOWNLOAD_DIR)/pocketsphinx-python.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ 'https://github.com/synesthesiam/pocketsphinx-python/releases/download/v1.0/pocketsphinx-python.tar.gz'
