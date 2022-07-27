# Standard Library Modules
import json
import logging
import os
import pathlib
import pickle
import sys
import tempfile
import zipfile

# Third-Party Modules
import PIL.Image as Image
import shutil
import yaml
import dotenv

# Local Modules
import utils.datetime_utils as datetime_utils

# Logging
def load_logger(script_name):
	logger = logging.getLogger(script_name)
	handler = logging.StreamHandler(stream=sys.stdout)
	handler.setLevel(logging.INFO)
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)

	return logger

def load_logger_to_file(script_name, base_log_dir):
	logging_dir = os.path.join(base_log_dir, 'monitoring/logs')
	print(logging_dir)
	check_dir_exists(logging_dir)

	filename = datetime_utils.get_datetime_filename(mantissa=True)
	log_filepath = os.path.join(logging_dir, filename + '.log').replace('\\', '/')

	logger = logging.getLogger(script_name)
	logging.basicConfig(filename=log_filepath,
						filemode='a',
						format='%(asctime)s %(levelname)-8s %(levelname)-8s %(name)-20s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d_%H:%M:%S')

	handler = logging.StreamHandler(stream=sys.stdout)
	handler.setLevel(logging.INFO)

	logger.addHandler(handler)
	logger.setLevel(logging.INFO)

	return logger

# File I/O Methods
def check_dir_exists(dir_path):
	if not os.path.isdir(dir_path):
		os.makedirs(dir_path)

def check_relative_path(content_path, use_dir_name=True):
	if not os.path.isabs(content_path):
		if use_dir_name:
			dir_name = os.path.dirname(__file__)
			content_path = os.path.join(dir_name, content_path)
		else:
			content_path = os.path.abspath(content_path)

	return content_path.replace('\\', '/')

def get_all_files(root_dir, file_ext=None):
	file_list = []
	for root, dirs, files in os.walk(root_dir):
		for file in sorted(files):
			if file_ext:
				if not isinstance(file_ext, str):
					for specExt in file_ext:
						if file.endswith(specExt):
							file_list.append(os.path.join(root, file))
							break
				else:
					if file.endswith(file_ext):
						file_list.append(os.path.join(root, file))
			else:
				file_list.append(os.path.join(root, file))

	return file_list


def get_dir_files_only(root_dir, file_ext=None):
	file_list = []
	for file in os.listdir(root_dir):
		if file_ext:
			if file.endswith(file_ext):
				file_list.append(os.path.join(root_dir, file))
		else:
			file_list.append(os.path.join(root_dir, file))

	return file_list


def get_sub_directories(parent_dir):
	dir_list = []
	for sub_dir in os.scandir(parent_dir):
		if sub_dir.is_dir():
			dir_list.append(sub_dir.path)

	return dir_list


def copy_all_files(target_dir, dest_dir, file_ext=None):
	files_list = get_all_files(root_dir=target_dir, file_ext=file_ext)

	for file_path in files_list:
		rel_path = remove_common_prefix_path(file_path, target_dir)
		dest_path = os.path.join(dest_dir, rel_path).replace('\\', '/')
		if not os.path.isdir(os.path.dirname(dest_path)):
			os.makedirs(os.path.dirname(dest_path))

		shutil.copyfile(file_path, dest_path)


def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        print(source, destination, archive_from, archive_to)
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)


def unzip_archive(zip_path):
	tmp_model_dir = tempfile.mkdtemp()
	with zipfile.ZipFile(zip_path, 'r') as zip_ref:
		zip_ref.extractall(path=tmp_model_dir)

	return os.path.join(tmp_model_dir, pathlib.Path(zip_path).stem)
	
def load_env_file(env_path):
	dotenv.load_dotenv(env_path)

def load_image(image_path):
	img = Image.open(image_path)
	return img

def is_legit_image_file(image_path):
	# img = Image.open(image_path)
	# return True
	try:
		img = Image.open(image_path)
		return True
	except:
		return False
	
def load_pickle(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f)

def load_yaml(file_path):
	stream = open(file_path, 'rb')
	yaml_dict = yaml.safe_load(stream)

	return yaml_dict


def remove_common_prefix_path(path, common_prefix):
	return os.path.relpath(path, common_prefix)

def write_yaml(yaml_dict, file_path):
	with open(file_path, 'w') as file:
		yaml.dump(yaml_dict, file, default_flow_style=False)


def write_json(json_dict, file_path):
	with open(file_path, "w") as file:
		json.dump(json_dict, file)