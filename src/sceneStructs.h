#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"
#include <glm/gtc/quaternion.hpp>

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

struct BoundingBox {
    glm::vec3 min;
    glm::vec3 max;
    bool valid = false;
    //BoundingBox();
    //BoundingBox(glm::vec3 min, glm::vec3 max);
    //BoundingBox getAABB(glm::mat4 m);
};

// glTF struct
struct GltfTexture {
    int width, height, size;
	int components; // e.g. 3 for RGB, 4 for RGBA
	unsigned char* imageData = nullptr;
};

struct GltfTextureInfo {
    int index{ -1 };    // required.
    int texCoord{ 0 };  // The set index of texture's TEXCOORD attribute used for
    // texture coordinate mapping.

    //Value extras;
    //ExtensionMap extensions;

    // Filled when SetStoreOriginalJSONForExtrasAndExtensions is enabled.
    //std::string extras_json_string;
    //std::string extensions_json_string;

    GltfTextureInfo() = default;
    //DEFAULT_METHODS(TextureInfo)
    //    bool operator==(const TextureInfo&) const;
};

//from tinygltf
struct GltfPbrMetallicRoughness {
	glm::vec3 baseColorFactor = glm::vec3(1.0f);  // here we use RGB, ignore A
    GltfTextureInfo baseColorTexture;
    float metallicFactor{ 1.0 };   // default 1
    float roughnessFactor{ 1.0 };  // default 1
    GltfTextureInfo metallicRoughnessTexture;

    //Value extras;
    //ExtensionMap extensions;

    // Filled when SetStoreOriginalJSONForExtrasAndExtensions is enabled.
    std::string extras_json_string;
    std::string extensions_json_string;

    GltfPbrMetallicRoughness() = default;
    //DEFAULT_METHODS(PbrMetallicRoughness)

    //    bool operator==(const PbrMetallicRoughness&) const;
};

// combine gltf and json material struct
struct Materialz
{
    int type;
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    bool isGltf = false;
    int texOffset = 0;

    struct TexCoordSets {
        uint8_t baseColor = 0;
        uint8_t metallicRoughness = 0;
        uint8_t specularGlossiness = 0;
        uint8_t normal = 0;
        uint8_t occlusion = 0;
        uint8_t emissive = 0;
    } texCoordSets;

	glm::vec3 baseColorFactor = glm::vec3(1.0f);
	int baseColorTexId = -1;
};
//struct GltfMaterial {
//    glm::vec4 baseColorFactor = glm::vec4(1.0f);
//    glm::vec4 emissiveFactor = glm::vec4(0.0f);
//    int baseColorTexId = -1;
//    int normalTexId = -1;
//    int metallicRoughnessTexId = -1;
//    int emissiveTexId = -1;
//};

struct GltfPrimitive {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;
    int materialId = -1;
    bool hasIndices = true;

    uint32_t firstTriangle = 0;
    uint32_t triangleCount = 0;
    BoundingBox bb;
    //void setBoundingBox(glm::vec3 min, glm::vec3 max);
};

struct GltfMesh {
	uint32_t primIndexStart;
	uint32_t primCount = 0;
    //BoundingBox bb;
    //BoundingBox aabb;
    glm::mat4 matrix;
    //glm::mat4 jointMatrix[MAX_NUM_JOINTS]{};
    //uint32_t jointcount{ 0 };
    uint32_t index;
    //void setBoundingBox(glm::vec3 min, glm::vec3 max);
};

struct GltfNode {
    GltfNode* parent;
    uint32_t index;
    std::vector<GltfNode*> children;
    glm::mat4 matrix;
    std::string name;
    GltfMesh* mesh;
    glm::vec3 translation{};
    glm::vec3 scale{ 1.0f };
    glm::quat rotation{};
    //BoundingBox bvh;
    //BoundingBox aabb;
    bool useCachedMatrix{ false };
    //glm::mat4 cachedLocalMatrix{ glm::mat4(1.0f) };
    //glm::mat4 cachedMatrix{ glm::mat4(1.0f) };
    //glm::mat4 localMatrix();
    //glm::mat4 getMatrix();
    //void update();
    ~GltfNode();
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv0;
};

struct LoaderInfo {
    uint32_t* indexBuffer;
    Vertex* vertexBuffer;
    size_t indexPos = 0;
    size_t vertexPos = 0;
};

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int meshid = -1;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct GltfVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct GltfTriangle {
    glm::vec3 v0, v1, v2;
    glm::vec3 n0, n1, n2;
    glm::vec2 uv0, uv1, uv2;
};

enum MaterialType {
  MATERIAL_DIFFUSE = 0,
  MATERIAL_SPECULAR = 1,
  MATERIAL_EMITTING = 2,
};

struct Cameraz
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Cameraz camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
};
