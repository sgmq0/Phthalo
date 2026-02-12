#include "stdafx.h"
#include "Camera.h"

Camera::Camera(XMFLOAT3 position, XMFLOAT3 up, float yaw, float pitch) :
    m_position(position),
    m_worldUp(up),
    m_yaw(yaw),
    m_pitch(pitch)
{
    updateCameraVectors();
}

void Camera::Update(float dt) {
    XMVECTOR pos     = XMLoadFloat3(&m_position);
    XMVECTOR front   = XMLoadFloat3(&m_front);
    XMVECTOR right   = XMLoadFloat3(&m_right);

    if (m_keys['W']) pos = XMVectorAdd(pos, XMVectorScale(front,  m_movementSpeed * dt));
    if (m_keys['S']) pos = XMVectorAdd(pos, XMVectorScale(front, -m_movementSpeed * dt));
    if (m_keys['D']) pos = XMVectorAdd(pos, XMVectorScale(right, -m_movementSpeed * dt));
    if (m_keys['A']) pos = XMVectorAdd(pos, XMVectorScale(right,  m_movementSpeed * dt));

    // vertical movement
    if (m_keys['E']) pos = XMVectorAdd(pos, XMVectorScale(XMLoadFloat3(&m_up),  m_movementSpeed * dt));
    if (m_keys['Q']) pos = XMVectorAdd(pos, XMVectorScale(XMLoadFloat3(&m_up), -m_movementSpeed * dt));

    XMStoreFloat3(&m_position, pos);
}

void Camera::OnKeyDown(UINT8 key) {
    m_keys[key] = true;
}

void Camera::OnKeyUp(UINT8 key) {
    m_keys[key] = false;
}

XMMATRIX Camera::GetViewMatrix() const {
    XMVECTOR pos   = XMLoadFloat3(&m_position);
    XMVECTOR front = XMLoadFloat3(&m_front);
    XMVECTOR up    = XMLoadFloat3(&m_up);
    return XMMatrixLookAtLH(pos, XMVectorAdd(pos, front), up);
}

XMMATRIX Camera::GetProjectionMatrix(float aspectRatio) const {
    return XMMatrixPerspectiveFovLH(
        XMConvertToRadians(m_zoom),
        aspectRatio,
        0.1f,
        1000.0f
    );
}

void Camera::updateCameraVectors() {
    XMFLOAT3 front;
    front.x = cos(XMConvertToRadians(m_yaw)) * cos(XMConvertToRadians(m_pitch));
    front.y = sin(XMConvertToRadians(m_pitch));
    front.z = sin(XMConvertToRadians(m_yaw)) * cos(XMConvertToRadians(m_pitch));

    XMVECTOR vFront = XMVector3Normalize(XMLoadFloat3(&front));
    XMStoreFloat3(&m_front, vFront);

    XMVECTOR vRight = XMVector3Normalize(XMVector3Cross(vFront, XMLoadFloat3(&m_worldUp)));
    XMStoreFloat3(&m_right, vRight);

    XMVECTOR vUp = XMVector3Normalize(XMVector3Cross(vRight, vFront));
    XMStoreFloat3(&m_up, vUp);
}