from setuptools import setup, find_packages

setup(
    name="nebula-architect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core Dependencies
        "torch>=2.0.0",
        "torch3d>=0.7.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pillow>=9.0.0",
        
        # 3D and Graphics
        "kaolin>=0.13.0",
        "open3d>=0.15.0",
        "three.js>=0.150.0",
        "pybullet>=3.2.5",
        
        # AI and Machine Learning
        "transformers>=4.30.0",
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "stable-diffusion-pytorch>=2.0.0",
        "clip @ git+https://github.com/openai/CLIP.git",
        "whisper>=1.0.0",
        "wav2vec2>=0.1.0",
        
        # Physics and Simulation
        "physx>=5.1.0",
        "ray[rllib]>=2.5.0",
        
        # Utilities
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "sounddevice>=0.4.6",
        "soundfile>=0.12.1",
    ],
    python_requires=">=3.9",
    author="NEBULA Architect Team",
    author_email="your.email@example.com",
    description="A revolutionary multimodal AI system that transforms spoken narrative descriptions into immersive 3D worlds",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nebula-architect",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            "nebula=src.main:main",
        ],
    },
) 