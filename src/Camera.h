#pragma once
#include "stdafx.h"

using namespace DirectX;

class Camera {
public:
    Camera(
        XMFLOAT3 position, 
        XMFLOAT3 up,
        float yaw,
        float pitch
    );

    void Update(float dt); // called in OnUpdate
    void OnKeyDown(UINT8 key);
    void OnKeyUp(UINT8 key);

    XMMATRIX GetViewMatrix() const;
    XMMATRIX GetProjectionMatrix(float aspectRatio) const;
    
    XMFLOAT3 m_position;
    XMFLOAT3 m_front;
    XMFLOAT3 m_up;
    XMFLOAT3 m_right;
    XMFLOAT3 m_worldUp;

    float m_yaw;
    float m_pitch;

    float m_movementSpeed = 3.0f;
    float m_mouseSensitivity = 0.002f;
    float m_zoom = 60.0f;

    // key state
    bool m_keys[256] = {};

private:
    void updateCameraVectors();
};
