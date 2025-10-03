#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"

using namespace tinygltf;
using namespace std;
using json = nlohmann::json;

Scenez::Scenez(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else if (ext == ".gltf")
    {
		loadFromGLTF(filename);
		return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scenez::loadFromGLTF(const std::string& gltfName)
{
    Model model;
    TinyGLTF loader;
    string err;
    string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName.c_str());
    if (!warn.empty())
    {
        cout << "Warn: " << warn << endl;
    }
    if (!err.empty())
    {
        cout << "Err: " << err << endl;
    }
    if (!ret)
    {
        cout << "Failed to parse glTF" << endl;
        return;
    }

    cout << "loaded glTF file has:\n"
        << model.accessors.size() << " accessors\n"
        << model.animations.size() << " animations\n"
        << model.buffers.size() << " buffers\n"
        << model.bufferViews.size() << " bufferViews\n"
        << model.materials.size() << " materials\n"
        << model.meshes.size() << " meshes\n"
        << model.nodes.size() << " nodes\n"
        << model.textures.size() << " textures\n"
        << model.images.size() << " images\n"
        << model.skins.size() << " skins\n"
        << model.samplers.size() << " samplers\n"
        << model.cameras.size() << " cameras\n"
        << model.scenes.size() << " scenes\n"
        << model.lights.size() << " lights\n";

	int texOffset = textures.size();
    loadGltfTextures(model);
	int matOffset = materials.size();
    loadGltfMaterial(model, texOffset);
	size_t vertexCount = 0;
	size_t indexCount = 0;
	const tinygltf::Scene& scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
    for (size_t i = 0; i < scene.nodes.size(); i++) {
        getNodeProps(model.nodes[scene.nodes[i]], model, vertexCount, indexCount);
    }
    cout << "index count: " << indexCount << "\n"
        << "vertex count: " << vertexCount << "\n";

    LoaderInfo loaderInfo;
    loaderInfo.vertexBuffer = new Vertex[vertexCount];
    loaderInfo.indexBuffer = new uint32_t[indexCount];
    for (size_t i = 0; i < scene.nodes.size(); i++) {
        const tinygltf::Node node = model.nodes[scene.nodes[i]];
        loadNode(nullptr, node, scene.nodes[i], model, loaderInfo, glm::mat4(1.0f), matOffset);
    }

	vertices.assign(loaderInfo.vertexBuffer, loaderInfo.vertexBuffer + vertexCount);
	indices.assign(loaderInfo.indexBuffer, loaderInfo.indexBuffer + indexCount);

    buildTriangleBuffer();

	cout << "loaded total " << meshes.size() << " meshes\n";
    for (size_t i = 0; i < meshes.size(); i++) {
        cout << "mesh " << i << " prim start: " << meshes[i].primIndexStart << " prim count: " << meshes[i].primCount << "\n";
	}
	cout << "loaded total " << primitives.size() << " primitives\n";
    for (size_t i = 0; i < primitives.size(); i++) {
        cout << "primitive " << i << " first index: " << primitives[i].firstIndex << " index count: " << primitives[i].indexCount 
            << " vertex count: " << primitives[i].vertexCount << " material id " << primitives[i].materialId << "\n"
            << " bbmin " << primitives[i].bb.min.x << " " << primitives[i].bb.min.y << " " << primitives[i].bb.min.z << "\n"
            << " bbmax " << primitives[i].bb.max.x << " " << primitives[i].bb.max.y << " " << primitives[i].bb.max.z << "\n";
    }
	cout << "loaded total " << triangles.size() << " triangles\n";
	cout << triangles[0].v0.x << " " << triangles[0].v0.y << " " << triangles[0].v0.z << "\n";
    cout << "loaded total " << textures.size() << " textures\n";
    cout << "loaded total " << materials.size() << " materials\n";
    delete[] loaderInfo.vertexBuffer;
    delete[] loaderInfo.indexBuffer;
}
void Scenez::loadGltfMaterial(const tinygltf::Model& model, const int& texOffset) {
    for (const tinygltf::Material& mat : model.materials) {
        Materialz newMat;

        newMat.type = 0;
		newMat.isGltf = true;
		newMat.texOffset = texOffset;

		newMat.baseColorFactor = glm::make_vec3(mat.pbrMetallicRoughness.baseColorFactor.data());
		newMat.baseColorTexId = mat.pbrMetallicRoughness.baseColorTexture.index;
		newMat.texCoordSets.baseColor = mat.pbrMetallicRoughness.baseColorTexture.texCoord;
        materials.push_back(newMat);
	}
}
void Scenez::loadGltfTextures(const tinygltf::Model& model) {

    for (const tinygltf::Texture& gltfTexture : model.textures) {
        GltfTexture newTexture;
        const tinygltf::Image& image = model.images[gltfTexture.source];
        newTexture.components = image.component;
        newTexture.width = image.width;
        newTexture.height = image.height;
        newTexture.size = image.component * image.width * image.height * sizeof(unsigned char);
        newTexture.imageData = new unsigned char[newTexture.size];
        memcpy(newTexture.imageData, image.image.data(), newTexture.size);
        textures.push_back(newTexture);
    }
}

void Scenez::getNodeProps(const tinygltf::Node& node, const tinygltf::Model& model, size_t& vertexCount, size_t& indexCount) {
    if (node.children.size() > 0) {
        for (size_t i = 0; i < node.children.size(); i++) {
            getNodeProps(model.nodes[node.children[i]], model, vertexCount, indexCount);
        }
    }
    if (node.mesh > -1) {
        const tinygltf::Mesh mesh = model.meshes[node.mesh];
        for (size_t i = 0; i < mesh.primitives.size(); i++) {
            auto& primitive = mesh.primitives[i];
            vertexCount += model.accessors[primitive.attributes.find("POSITION")->second].count;
            if (primitive.indices > -1) {
                indexCount += model.accessors[primitive.indices].count;
            }
        }
    }
}

// modified from https://github.com/SaschaWillems/Vulkan-glTF-PBR
void Scenez::loadNode(GltfNode* parent, const tinygltf::Node& node, uint32_t nodeIndex, const tinygltf::Model& model, LoaderInfo& loaderInfo, glm::mat4 pTransform, int matOffset) {

    GltfNode* newNode = new GltfNode{};
    newNode->index = nodeIndex;
    newNode->parent = parent;
    newNode->name = node.name;
    newNode->matrix = glm::mat4(1.0f);

    // Generate local node matrix
    glm::vec3 translation = glm::vec3(0.0f);
    if (node.translation.size() == 3) {
        translation = glm::make_vec3(node.translation.data());
        newNode->translation = translation;
    }
    glm::mat4 rotation = glm::mat4(1.0f);
    if (node.rotation.size() == 4) {
        glm::quat q = glm::make_quat(node.rotation.data());
        newNode->rotation = q;
    }
    glm::vec3 scale = glm::vec3(1.0f);
    if (node.scale.size() == 3) {
        scale = glm::make_vec3(node.scale.data());
        newNode->scale = scale;
    }
    if (node.matrix.size() == 16) {
        newNode->matrix = glm::make_mat4x4(node.matrix.data());
    }
    else {
        newNode->matrix = utilityCore::buildTransformationMatrixQuat(newNode->translation, newNode->rotation, newNode->scale);
    }
    // compute global transform
	glm::mat4 globalTransform = newNode->matrix * pTransform;

    if (node.children.size() > 0) {
        for (size_t i = 0; i < node.children.size(); i++) {
            loadNode(newNode, model.nodes[node.children[i]], node.children[i], model, loaderInfo, globalTransform, matOffset);
        }
    }
    if (node.mesh > -1) {
        const tinygltf::Mesh mesh = model.meshes[node.mesh];
		Geom newGeom;
		newGeom.type = GeomType::MESH;
		newGeom.transform = globalTransform;
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        newGeom.materialid = 3; // we are not using this

		newGeom.meshid = meshes.size() + node.mesh;
        geoms.push_back(newGeom);

		//cout << "new mesh geom initializing: " << geoms.size() - 1 << " mesh id " << newGeom.meshid << " name: " << node.name << "\n";
		GltfMesh newMesh;
		newMesh.index = node.mesh;
		newMesh.matrix = newGeom.transform;
		newMesh.primIndexStart = primitives.size();
		newMesh.primCount = static_cast<uint32_t>(mesh.primitives.size());

        for (size_t j = 0; j < mesh.primitives.size(); j++) {
            auto& primitive = mesh.primitives[j];
            uint32_t vertexStart = static_cast<uint32_t>(loaderInfo.vertexPos);
            uint32_t indexStart = static_cast<uint32_t>(loaderInfo.indexPos);
            uint32_t indexCount = 0;
            uint32_t vertexCount = 0;

            glm::vec3 posMin{};
            glm::vec3 posMax{};

            bool hasIndices = primitive.indices > -1;

            const float* bufferPos = nullptr;
            const float* bufferNormals = nullptr;
            const float* bufferTexCoordSet0 = nullptr;

            int posByteStride;
            int normByteStride;
            int uv0ByteStride;
			// POSITION is required
            assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
            const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
            bufferPos = reinterpret_cast<const float*>(&(model.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));
            posMin = glm::vec3(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);
            posMax = glm::vec3(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);
            vertexCount = static_cast<uint32_t>(posAccessor.count);
            posByteStride = posAccessor.ByteStride(posView) ? (posAccessor.ByteStride(posView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);

            // Normal
            if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                const tinygltf::Accessor& normAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
                const tinygltf::BufferView& normView = model.bufferViews[normAccessor.bufferView];
                bufferNormals = reinterpret_cast<const float*>(&(model.buffers[normView.buffer].data[normAccessor.byteOffset + normView.byteOffset]));
                normByteStride = normAccessor.ByteStride(normView) ? (normAccessor.ByteStride(normView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
            }

            //TODO: for now we only consider "TEXCOORD_0"
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                const tinygltf::Accessor& uvAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
                const tinygltf::BufferView& uvView = model.bufferViews[uvAccessor.bufferView];
                bufferTexCoordSet0 = reinterpret_cast<const float*>(&(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
                uv0ByteStride = uvAccessor.ByteStride(uvView) ? (uvAccessor.ByteStride(uvView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
            }

            for (size_t v = 0; v < posAccessor.count; v++) {
                Vertex& vert = loaderInfo.vertexBuffer[loaderInfo.vertexPos];
                vert.pos = glm::make_vec3(&bufferPos[v * posByteStride]);
                vert.normal = glm::normalize(glm::vec3(bufferNormals ? glm::make_vec3(&bufferNormals[v * normByteStride]) : glm::vec3(0.0f)));
                vert.uv0 = bufferTexCoordSet0 ? glm::make_vec2(&bufferTexCoordSet0[v * uv0ByteStride]) : glm::vec2(0.0f);

                loaderInfo.vertexPos++;
            }

            //indices
            if (hasIndices) {
				const tinygltf::Accessor &accessor = model.accessors[primitive.indices > -1 ? primitive.indices : 0];
				const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
				const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

				indexCount = static_cast<uint32_t>(accessor.count);
				const void *dataPtr = &(buffer.data[accessor.byteOffset + bufferView.byteOffset]);

				switch (accessor.componentType) {
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
					const uint32_t *buf = static_cast<const uint32_t*>(dataPtr);
					for (size_t index = 0; index < accessor.count; index++) {
						loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
						loaderInfo.indexPos++;
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
					const uint16_t *buf = static_cast<const uint16_t*>(dataPtr);
					for (size_t index = 0; index < accessor.count; index++) {
						loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
						loaderInfo.indexPos++;
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
					const uint8_t *buf = static_cast<const uint8_t*>(dataPtr);
					for (size_t index = 0; index < accessor.count; index++) {
						loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
						loaderInfo.indexPos++;
					}
					break;
				}
				default:
					std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
					return;
				}
			}

            GltfPrimitive newPrimitive;
            newPrimitive.firstIndex = indexStart;
            newPrimitive.indexCount = indexCount;
            newPrimitive.vertexCount = vertexCount;
            newPrimitive.materialId = matOffset + primitive.material;
			//newPrimitive.bb.min = posMin;
            //newPrimitive.bb.max = posMax;
			//newPrimitive.bb.valid = true;

			primitives.emplace_back(newPrimitive);
        }

        meshes.emplace_back(newMesh);
		//cout << "loaded mesh " << meshes.size() - 1 << " prim start idx: " << newMesh.primIndexStart << " prim count: " << newMesh.primCount << "\n";
    }
}

void Scenez::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Materialz newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
          newMaterial.type = MATERIAL_DIFFUSE;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
          newMaterial.type = MATERIAL_EMITTING;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
          newMaterial.type = MATERIAL_SPECULAR;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
	const auto& gltfData = data["GLTF"];
    for (const auto& p : gltfData)
    {
        loadFromGLTF(p["PATH"]);
    }
    cout << "loaded total " << geoms.size() << " geoms\n";
    for (int i = 0; i < geoms.size(); i++) {
        cout << "geom " << i << " type " << geoms[i].type << " mesh id " << geoms[i].meshid << " material id " << geoms[i].materialid << "\n";
    }


    const auto& cameraData = data["Camera"];
    Cameraz& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scenez::buildTriangleBuffer() {
    triangles.clear();

    for (auto& prim : primitives) {
        glm::vec3 minBB(FLT_MAX);
        glm::vec3 maxBB(-FLT_MAX);

        if (!prim.hasIndices || prim.indexCount < 3) continue;

        prim.firstTriangle = static_cast<uint32_t>(triangles.size());
        prim.triangleCount = prim.indexCount / 3;

        uint32_t start = prim.firstIndex;
        uint32_t end = start + prim.indexCount;

        for (uint32_t k = start; k + 2 < end; k += 3) {
            uint32_t i0 = indices[k + 0];
            uint32_t i1 = indices[k + 1];
            uint32_t i2 = indices[k + 2];

            const Vertex& v0 = vertices[i0];
            const Vertex& v1 = vertices[i1];
            const Vertex& v2 = vertices[i2];

            GltfTriangle tri;
            tri.v0 = v0.pos;
            tri.v1 = v1.pos;
            tri.v2 = v2.pos;

            tri.n0 = v0.normal;
            tri.n1 = v1.normal;
            tri.n2 = v2.normal;

            tri.uv0 = v0.uv0;
            tri.uv1 = v1.uv0;
            tri.uv2 = v2.uv0;

            triangles.push_back(tri);

            minBB = glm::min(minBB, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
            maxBB = glm::max(maxBB, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
        }

		prim.bb.min = minBB;
        prim.bb.max = maxBB;
		prim.bb.valid = true;
    }
}