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
    
    XMFLOAT3 m_position;
    XMFLOAT3 m_front;
    XMFLOAT3 m_up;
    XMFLOAT3 m_right;
    XMFLOAT3 m_worldUp;

    float m_yaw;
    float m_pitch;

    float m_movementSpeed;
    float m_mouseSensitivity;
    float m_zoom;

private:
    void updateCameraVectors();
};
