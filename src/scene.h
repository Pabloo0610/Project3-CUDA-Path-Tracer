#pragma once

#include "sceneStructs.h"
#include <vector>
#include "tiny_gltf.h"

class Scenez
{
private:
    void loadFromJSON(const std::string& jsonName);
	void loadFromGLTF(const std::string& gltfName);
    void getNodeProps(const tinygltf::Node& node, const tinygltf::Model& model, size_t& vertexCount, size_t& indexCount);
    void loadGltfTextures(const tinygltf::Model& model);
	void loadGltfMaterial(const tinygltf::Model& model, const int& texOffset);
	void loadNode(GltfNode* parent, const tinygltf::Node& node, uint32_t nodeIndex, const tinygltf::Model& model, LoaderInfo& loaderInfo, glm::mat4 p_transform, int matOffset);
	void buildTriangleBuffer();

public:
    Scenez(std::string filename);

	std::vector<GltfTexture> textures;
	std::vector<GltfMesh> meshes;
	std::vector<GltfPrimitive> primitives;
	std::vector<GltfTriangle> triangles;
    std::vector<Geom> geoms;
    std::vector<Materialz> materials;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
    RenderState state;
};
